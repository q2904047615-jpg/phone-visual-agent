from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import threading
import time
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import replace
from typing import Any, Callable, Iterator

from PIL import Image, ImageChops, ImageFilter

from element_geometry_audit import (
    ElementGeometryAuditError,
    build_candidate_crop_transform,
    build_input_candidate_crop_transform,
    build_literal_candidate_crop_transform,
    element_geometry_audit_prompt,
    select_unique_audited_geometry,
)
from observation_images import (
    VisualObstruction,
    consensus_top_edge_obstructions,
    map_roi_bounds_to_full,
    measure_frame_sharpness,
    measure_local_stability,
)
from orientation_safety import (
    ORIENTATION_AUDIT_PROTOCOL_VERSION,
    OrientationCredential,
    OrientationSafetyError,
    _mint_audited_credential,
)
from qwen_runtime_errors import (
    FORMAT_ERROR_TYPES,
    classify_qwen_error,
    failure_diagnostics,
)
from ocr_runtime import find_text as find_ocr_text, recognize as recognize_ocr
from robot_core import WorkflowNotReady, qwerty_keyboard_config_from_anchors
from ui_scene import (
    ALLOWED_ROLES,
    CAMERA_LAYOUT_ORIENTATIONS,
    CameraAlignmentFacts,
    MIN_TARGET_CONFIDENCE,
    PHONE_CONTENT_ROTATIONS,
    SystemUIFacts,
    UI_SCENE_PROTOCOL_VERSION,
    UIScene,
    UISceneError,
    camera_alignment_evidence_is_safe,
)
from vision_agent import VisionAgentError, _extract_json_object, _image_data_url
from vision_model_config import public_model_identity
from verified_text_transaction import (
    VerifiedTextTransactionError,
    plan_next_verified_input,
)
from input_value_lineage import (
    PENDING_INPUT_LINEAGE_SOURCES,
    TypedInputLineage,
    TypedInputLineageStore,
)
from system_navigation_privacy import (
    SYSTEM_NAVIGATION_PRIVACY_VIEW_VERSION,
    privacy_minimized_system_navigation_view,
)


GENERIC_SCENE_OBSERVER_VERSION = "2026-08-24-generic-scene-observer-v70"
SINGLE_STEP_SCENE_OBSERVER_VERSION = "2026-08-24-single-step-scene-observer-v1"
SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION = (
    "2026-08-24-single-step-qwen-observation-v1"
)
POST_NAVIGATION_RESULT_OBSERVATION_PHASE = "verified_navigation_result_v1"
POST_NAVIGATION_RESULT_OBJECTIVE = "观察本次导航后的当前稳定画面"
POST_NAVIGATION_RESULT_COMPLETION_CONDITIONS = ["当前稳定结果画面已被重新观察"]
TARGETED_SCENE_DELTA_PROTOCOL_VERSION = "2026-08-17-targeted-scene-delta-v1"
FOREGROUND_APP_IDENTITY_AUDIT_VERSION = (
    "2026-08-18-foreground-app-identity-audit-v1"
)
INPUT_STRUCTURE_AUDIT_VERSION = "2026-08-23-input-structure-audit-v9"
SYSTEM_UI_AUDIT_VERSION = "2026-08-14-system-ui-audit-v1"
ICON_CLUSTER_AUDIT_VERSION = "2026-08-15-icon-cluster-audit-v1"
COMPACT_OUTPUT_TOKENS = 2600
# Targeted delta permits the same maximum element count as compact observation.
# Its output budget must therefore cover the same strict worst-case structure.
TARGETED_OUTPUT_TOKENS = COMPACT_OUTPUT_TOKENS
FOREGROUND_APP_IDENTITY_AUDIT_TOKENS = 300
# The strict input schema can include application inputs, IME candidates and a
# complete keyboard structure in one response.  It needs the same ceiling as a
# compact scene; actual billing still follows emitted tokens, not this ceiling.
INPUT_STRUCTURE_AUDIT_TOKENS = COMPACT_OUTPUT_TOKENS
SYSTEM_UI_AUDIT_TOKENS = 600
ICON_CLUSTER_AUDIT_TOKENS = 700
ELEMENT_GEOMETRY_AUDIT_TOKENS = 500
ORIENTATION_AUDIT_TOKENS = 500
SINGLE_STEP_OUTPUT_TOKENS = 5200
MIN_SYSTEM_UI_AUDIT_CONFIDENCE = 0.80
OBSERVATION_TIMEOUT_SECONDS = 60.0
MAX_COMPACT_ELEMENTS = 12
AUDITED_SOFT_KEYBOARD_HIDDEN_EVIDENCE = "输入结构只读审计确认软键盘不可见"

STAGE_LABELS = {
    "idle": "空闲",
    "checking_stability": "检查画面稳定性",
    "waiting_single_step_observation": "等待千问单步完整观察",
    "parsing_single_step_observation": "解析单步完整观察",
    "waiting_compact_observation": "等待千问快速观察",
    "parsing_compact_observation": "解析快速观察结果",
    "waiting_compact_retry": "等待千问修正观察格式",
    "parsing_compact_retry": "解析修正结果",
    "waiting_targeted_refinement": "等待千问目标精查",
    "parsing_targeted_refinement": "解析目标精查结果",
    "waiting_foreground_app_identity_audit": "等待前台应用身份只读审计",
    "parsing_foreground_app_identity_audit": "解析前台应用身份只读审计",
    "waiting_icon_cluster_audit": "等待图标簇只读审计",
    "parsing_icon_cluster_audit": "解析图标簇只读审计",
    "waiting_icon_cluster_localization": "等待图标簇局部定位复核",
    "parsing_icon_cluster_localization": "解析图标簇局部定位复核",
    "waiting_element_geometry_audit": "等待目标元素几何审计",
    "parsing_element_geometry_audit": "解析目标元素几何审计",
    "waiting_input_structure_audit": "等待输入结构只读审计",
    "parsing_input_structure_audit": "解析输入结构只读审计",
    "waiting_system_ui_audit": "等待系统界面只读审计",
    "parsing_system_ui_audit": "解析系统界面只读审计",
    "waiting_system_ui_audit_retry": "等待系统界面审计格式修正",
    "waiting_orientation_audit": "等待独立方向只读审计",
    "waiting_orientation_audit_retry": "等待独立方向证据格式重审",
    "parsing_orientation_audit": "解析独立方向只读审计",
    "completed": "观察完成",
    "failed": "观察安全停止",
}


def _stable_ocr_literal_bounds(
    frames: tuple[Image.Image, ...] | list[Image.Image],
    label: str,
    *,
    ocr_recognizer: Any = recognize_ocr,
    ocr_finder: Any = find_ocr_text,
) -> tuple[float, float, float, float] | None:
    """Return a unique three-frame literal-text box, or fail closed."""

    literal = str(label or "").strip()
    frame_list = list(frames)[-3:]
    if not literal or len(frame_list) != 3:
        return None
    matches = []
    try:
        for frame in frame_list:
            found = list(
                ocr_finder(
                    ocr_recognizer(frame.convert("RGB"), "zh-Hans-CN", scale=3.0),
                    literal,
                )
            )
            if len(found) != 1:
                return None
            match = found[0]
            if (
                match.width <= 0
                or match.height <= 0
                or match.left < 0
                or match.top < 0
                or match.left + match.width > frame.width
                or match.top + match.height > frame.height
            ):
                return None
            matches.append(match)
    except Exception:
        return None
    centers = [match.center for match in matches]
    if (
        max(point[0] for point in centers) - min(point[0] for point in centers) > 8
        or max(point[1] for point in centers) - min(point[1] for point in centers) > 8
    ):
        return None
    width, height = frame_list[-1].size
    left = float(statistics.median(match.left for match in matches))
    top = float(statistics.median(match.top for match in matches))
    right = float(
        statistics.median(match.left + match.width for match in matches)
    )
    bottom = float(
        statistics.median(match.top + match.height for match in matches)
    )
    return (left / width, top / height, right / width, bottom / height)


def _can_use_stable_ocr_literal_bounds(role: str, label: str) -> bool:
    """Limit literal OCR snapping to text-bearing selector roles."""

    return role in {"text", "button", "tab", "list_item"} and bool(
        str(label or "").strip()
    )


def _geometry_evidence_literal_labels(
    scene: UIScene,
    *,
    target_element_id: str,
    evidence: str,
) -> tuple[str, ...]:
    """Expose only literal labels that the selected evidence actually cites."""

    target = scene.get_element(target_element_id)
    selected: list[str] = []
    target_label = str(target.label or "").strip()
    if target_label:
        selected.append(target_label)
    evidence_text = str(evidence or "")
    for item in scene.elements:
        label = str(item.label or "").strip()
        if (
            label
            and label in evidence_text
            and label not in selected
        ):
            selected.append(label)
    # Do not truncate a genuinely dense evidence statement. Passing the full
    # set lets the strict geometry contract reject it instead of silently
    # changing which visible words are considered trustworthy.
    return tuple(selected)


class GenericSceneObserver:
    """Qwen reports the current scene; it never chooses or executes actions."""

    def __init__(
        self,
        provider: Any,
        *,
        input_lineage_store: TypedInputLineageStore | None = None,
        qwerty_row_snapper: Callable[
            [list[Image.Image] | tuple[Image.Image, ...], dict[str, Any]],
            dict[str, list[int]] | None,
        ]
        | None = None,
    ) -> None:
        self.provider = provider
        self.input_lineage_store = input_lineage_store
        self.qwerty_row_snapper = qwerty_row_snapper
        self.last_raw_response = ""
        self.last_diagnostics: dict[str, Any] = {}
        self._stage_lock = threading.RLock()
        self._current_stage = "idle"
        self._last_stage = "idle"
        self.last_orientation_audit_diagnostics: dict[str, Any] = {}
        self.last_geometry_audit_diagnostics: dict[str, Any] = {}
        self.supports_typed_input_continuation = True
        self._observation_cache_lock = threading.RLock()
        self._observation_cache: OrderedDict[str, UIScene] = OrderedDict()
        self._observation_cache_limit = 32

    def audit_element_geometry(
        self,
        *,
        frames: list[Image.Image] | tuple[Image.Image, ...],
        scene: UIScene,
        element_ids: tuple[str, ...] | list[str],
    ) -> UIScene:
        """Replace only selected bounds after strict one-crop localization."""

        self.last_geometry_audit_diagnostics = {}
        scene.validate()
        requested_ids = tuple(str(value or "").strip() for value in element_ids)
        if not 1 <= len(requested_ids) <= 2 or any(not value for value in requested_ids):
            raise ElementGeometryAuditError(
                "一次几何审计只接受1个目标，或拖动动作的2个端点。"
            )
        if len(requested_ids) != len(set(requested_ids)):
            raise ElementGeometryAuditError("几何审计目标 element_id 重复。")
        frame_list = list(frames)
        if len(frame_list) < 4:
            raise ElementGeometryAuditError("几何审计至少需要4帧稳定画面。")
        stability = measure_local_stability(
            frame_list,
            allow_leading_outlier=True,
        )
        if not stability.stable:
            raise ElementGeometryAuditError("几何审计画面未通过本地稳定性检查。")
        tail_start = max(0, len(frame_list) - min(3, len(frame_list)))
        matching_indices = tuple(
            index
            for index in range(tail_start, len(frame_list))
            if _local_frame_fingerprint(frame_list[index].convert("RGB"))
            == scene.fingerprint
        )
        if not matching_indices:
            raise ElementGeometryAuditError(
                "几何审计帧与可信场景 fingerprint 不一致。"
            )
        selected_index = max(
            matching_indices,
            key=lambda index: measure_frame_sharpness(frame_list[index]),
        )
        frame = frame_list[selected_index].convert("RGB")
        replacements: dict[str, tuple[float, float, float, float]] = {}
        audit_records: list[dict[str, Any]] = []
        for element_id in requested_ids:
            element = scene.get_element(element_id)
            literal_selector = _can_use_stable_ocr_literal_bounds(
                element.role,
                element.label,
            )
            if literal_selector:
                transform = build_literal_candidate_crop_transform(
                    frame.size, element.bounds
                )
                crop_profile = "literal_selector"
            elif element.role == "input":
                transform = build_input_candidate_crop_transform(
                    frame.size, element.bounds
                )
                crop_profile = "input_structural_context"
            else:
                transform = build_candidate_crop_transform(
                    frame.size, element.bounds
                )
                crop_profile = "broad_structural"
            source_digest = hashlib.sha256(
                frame.tobytes()
                + element.element_id.encode("utf-8")
                + element.label.encode("utf-8")
                + element.role.encode("utf-8")
            ).hexdigest()
            source_ref = f"geom-{source_digest[:24]}"
            prompt = None
            selected_evidence = ""
            visible_literal_labels: tuple[str, ...] = ()
            for evidence in element.evidence:
                try:
                    evidence_literal_labels = _geometry_evidence_literal_labels(
                        scene,
                        target_element_id=element_id,
                        evidence=evidence,
                    )
                    prompt = element_geometry_audit_prompt(
                        source_ref=source_ref,
                        literal_label=element.label,
                        visual_role=element.role,
                        visible_evidence=evidence,
                        visible_literal_labels=evidence_literal_labels,
                    )
                    selected_evidence = evidence
                    visible_literal_labels = evidence_literal_labels
                    break
                except ElementGeometryAuditError:
                    continue
            if prompt is None:
                raise ElementGeometryAuditError(
                    f"目标 {element_id} 缺少可用于独立几何审计的原始可见证据。"
                )
            crop = transform.crop(frame)
            self._set_stage("waiting_element_geometry_audit")
            scope_factory = getattr(self.provider, "call_scope", None)
            scope = (
                scope_factory(
                    stage="element_geometry_audit",
                    fingerprint=scene.fingerprint,
                )
                if callable(scope_factory)
                else nullcontext()
            )
            with scope:
                raw = self._provider_chat(
                    [
                        _json_only_system_message(),
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": _image_data_url(crop)},
                                },
                            ],
                        },
                    ],
                    max_tokens=ELEMENT_GEOMETRY_AUDIT_TOKENS,
                    response_format={"type": "json_object"},
                )
            self.last_raw_response = raw
            self._set_stage("parsing_element_geometry_audit")
            try:
                audited = select_unique_audited_geometry(
                    raw,
                    expected_source_ref=source_ref,
                    expected_label=element.label,
                    expected_role=element.role,
                    transform=transform,
                    visible_literal_labels=visible_literal_labels,
                )
            except Exception as exc:
                self.last_geometry_audit_diagnostics = {
                    "scene_fingerprint": scene.fingerprint,
                    "element_id": element_id,
                    "source_ref": source_ref,
                    "json_mode_requested": True,
                    "response_length": len(raw),
                    "response_text": raw[:4096],
                    "audit_accepted": False,
                    "error_type": classify_qwen_error(exc, raw_response=raw),
                }
                raise
            snapped_input_bounds = None
            if element.role == "input":
                snapped_input_bounds = _snap_audited_input_to_local_border(
                    frame,
                    transform=transform,
                    rough_bounds=element.bounds,
                    audited_bounds=audited.full_bounds,
                )
            ocr_literal_bounds = None
            if literal_selector:
                ocr_literal_bounds = _stable_ocr_literal_bounds(
                    [frame_list[index] for index in matching_indices],
                    element.label,
                )
            replacements[element_id] = (
                ocr_literal_bounds or snapped_input_bounds or audited.full_bounds
            )
            audit_records.append(
                {
                    "element_id": element_id,
                    "source_ref": source_ref,
                    "role": element.role,
                    "label": element.label,
                    "visible_evidence": selected_evidence,
                    "pixel_bounds": list(transform.pixel_bounds),
                    "crop_profile": crop_profile,
                    "local_bounds": list(audited.local_bounds),
                    "full_bounds": list(audited.full_bounds),
                    "local_border_snap_used": snapped_input_bounds is not None,
                    "local_ocr_literal_snap_used": ocr_literal_bounds is not None,
                    "ocr_literal_full_bounds": (
                        list(ocr_literal_bounds)
                        if ocr_literal_bounds is not None
                        else None
                    ),
                    "snapped_full_bounds": (
                        list(snapped_input_bounds)
                        if snapped_input_bounds is not None
                        else None
                    ),
                    "confidence": audited.confidence,
                }
            )
        audited_scene = replace(
            scene,
            elements=tuple(
                replace(
                    element,
                    bounds=replacements[element.element_id],
                    states={
                        **element.states,
                        # These private facts are minted only after the strict
                        # one-crop parser has proved one complete control.  The
                        # model-authored scene path strips both keys.
                        "fully_visible": True,
                        "independent_geometry_verified": True,
                        "geometry_audit_source": "element_geometry_audit",
                    },
                )
                if element.element_id in replacements
                else element
                for element in scene.elements
            ),
        )
        audited_scene.validate()
        self.last_geometry_audit_diagnostics = {
            "scene_fingerprint": scene.fingerprint,
            "selected_frame_index": selected_index,
            "audits": audit_records,
        }
        self._set_stage("completed")
        return audited_scene

    def audit_camera_alignment(
        self,
        *,
        frames: list[Image.Image],
        device_id: str,
        scene_fingerprint: str,
    ) -> OrientationCredential:
        """Mint action authority from a separate, read-only model response."""

        return self._audit_camera_alignment(
            frames=frames,
            device_id=device_id,
            scene_fingerprint=scene_fingerprint,
            privacy_minimized=False,
        )

    def audit_coordinate_free_system_navigation_alignment(
        self,
        *,
        frames: list[Image.Image],
        device_id: str,
        scene_fingerprint: str,
    ) -> OrientationCredential:
        """Audit Home orientation without disclosing unrelated App content."""

        return self._audit_camera_alignment(
            frames=frames,
            device_id=device_id,
            scene_fingerprint=scene_fingerprint,
            privacy_minimized=True,
        )

    def _audit_camera_alignment(
        self,
        *,
        frames: list[Image.Image],
        device_id: str,
        scene_fingerprint: str,
        privacy_minimized: bool,
    ) -> OrientationCredential:

        self.last_orientation_audit_diagnostics = {}
        if len(frames) < 4:
            raise VisionAgentError("方向独立审计至少需要4帧。")
        stability = measure_local_stability(frames, allow_leading_outlier=True)
        if not stability.stable:
            raise VisionAgentError("方向独立审计的本地帧不稳定。")
        tail_start = max(0, len(frames) - min(3, len(frames)))
        scores = [measure_frame_sharpness(item) for item in frames]
        selected_index = max(
            range(tail_start, len(frames)), key=scores.__getitem__
        )
        frame = frames[selected_index].convert("RGB")
        local_fingerprint = _local_frame_fingerprint(frame)

        model_frame = (
            privacy_minimized_system_navigation_view(frame)
            if privacy_minimized
            else frame
        )
        images = (
            model_frame,
            model_frame.transpose(Image.Transpose.ROTATE_90),
            model_frame.transpose(Image.Transpose.ROTATE_270),
        )
        image_roles = (
            "IMAGE 1 - CLASSIFICATION TARGET - ORIGINAL STABLE FRAME",
            "IMAGE 2 - REFERENCE ONLY - IMAGE 1 ROTATED 90 DEGREES",
            "IMAGE 3 - REFERENCE ONLY - IMAGE 1 ROTATED 270 DEGREES",
        )
        content: list[dict[str, Any]] = [
            {"type": "text", "text": _orientation_audit_prompt()}
        ]
        for role, item in zip(image_roles, images):
            content.extend(
                (
                    {"type": "text", "text": role},
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(item)},
                    },
                )
            )
        self._set_stage("waiting_orientation_audit")
        raw = ""
        model_calls = 0
        try:
            model_calls += 1
            scope_factory = getattr(self.provider, "call_scope", None)
            scope = (
                scope_factory(
                    stage="orientation_audit",
                    fingerprint=scene_fingerprint,
                )
                if callable(scope_factory)
                else nullcontext()
            )
            with scope:
                raw = self._provider_chat(
                    [
                        _json_only_system_message(),
                        {"role": "user", "content": content},
                    ],
                    max_tokens=ORIENTATION_AUDIT_TOKENS,
                )
            self._set_stage("parsing_orientation_audit")
            credential = _credential_from_orientation_audit(
                raw=raw,
                device_id=device_id,
                scene_fingerprint=scene_fingerprint,
                frame=frame,
            )
            credential.assert_authorizes(
                device_id=device_id,
                scene_fingerprint=scene_fingerprint,
                frame_size=tuple(frame.size),
            )
            self.last_orientation_audit_diagnostics = {
                "audit_version": ORIENTATION_AUDIT_PROTOCOL_VERSION,
                "model_calls": model_calls,
                "image_count": 3,
                "cache_hit": False,
                "selected_frame_index": selected_index,
                "frame_size": list(frame.size),
                "frame_fingerprint": local_fingerprint,
                "confidence": float(credential.confidence),
                "response_payload": _orientation_audit_diagnostic_payload(raw),
                "audit_accepted": True,
                "retry_used": False,
                "privacy_view_version": (
                    SYSTEM_NAVIGATION_PRIVACY_VIEW_VERSION
                    if privacy_minimized
                    else None
                ),
            }
            return credential
        except Exception as exc:
            self.last_orientation_audit_diagnostics = {
                "audit_version": ORIENTATION_AUDIT_PROTOCOL_VERSION,
                "model_calls": model_calls,
                "image_count": 3,
                "cache_hit": False,
                "selected_frame_index": selected_index,
                "frame_size": list(frame.size),
                "frame_fingerprint": local_fingerprint,
                "response_payload": _orientation_audit_diagnostic_payload(raw),
                "audit_accepted": False,
                "retry_used": False,
                "privacy_view_version": (
                    SYSTEM_NAVIGATION_PRIVACY_VIEW_VERSION
                    if privacy_minimized
                    else None
                ),
                "error_type": classify_qwen_error(exc, raw_response=raw),
            }
            if isinstance(exc, VisionAgentError):
                raise
            if isinstance(exc, (OrientationSafetyError, UISceneError, ValueError)):
                raise VisionAgentError(f"方向独立审计失败：{exc}") from exc
            raise
        finally:
            self._set_stage("idle")

    def _set_stage(self, stage: str) -> None:
        with self._stage_lock:
            self._current_stage = stage
            if stage != "idle":
                self._last_stage = stage

    def status(self) -> dict[str, Any]:
        value = dict(self.provider.status())
        with self._stage_lock:
            current_stage = self._current_stage
            last_stage = self._last_stage
        value.update(
            {
                "observer_version": GENERIC_SCENE_OBSERVER_VERSION,
                "scene_protocol": UI_SCENE_PROTOCOL_VERSION,
                "model_role": "generic_observation_only",
                "supported_app_scope": "dynamic",
                "hardware_actions_enabled": False,
                "current_stage": current_stage,
                "current_stage_label": STAGE_LABELS.get(current_stage, current_stage),
                "last_stage": last_stage,
                "last_stage_label": STAGE_LABELS.get(last_stage, last_stage),
                "compact_output_tokens": COMPACT_OUTPUT_TOKENS,
                "targeted_delta_protocol": TARGETED_SCENE_DELTA_PROTOCOL_VERSION,
                "targeted_output_tokens": TARGETED_OUTPUT_TOKENS,
                "observation_timeout_seconds": OBSERVATION_TIMEOUT_SECONDS,
                "max_compact_elements": MAX_COMPACT_ELEMENTS,
                "last_scene_enum_values": dict(
                    self.last_diagnostics.get("scene_enum_values") or {}
                ),
                "last_input_structure_shape": dict(
                    self.last_diagnostics.get("input_structure_shape") or {}
                ),
                "last_orientation_audit_diagnostics": dict(
                    self.last_orientation_audit_diagnostics
                ),
                "last_geometry_audit_diagnostics": dict(
                    self.last_geometry_audit_diagnostics
                ),
            }
        )
        return value

    def observe(
        self,
        *,
        frames: list[Image.Image],
        goal_context: dict[str, Any] | None = None,
        device_id: str | None = None,
        input_lineage_override: TypedInputLineage | None = None,
        prior_scene: UIScene | None = None,
    ) -> UIScene:
        self.last_raw_response = ""
        model_identity = public_model_identity(self.provider.status())
        self.last_diagnostics = {"vision_model": model_identity}
        self._set_stage("checking_stability")
        started = time.perf_counter()
        model_calls = 0
        compact_retry_used = False
        format_retry_used = False
        local_structural_repair_used = False
        targeted_refinement_used = False
        foreground_app_identity_audit_used = False
        foreground_app_identity_audit_confidence: float | None = None
        foreground_app_identity_audit_evidence: tuple[str, ...] = ()
        compact_geometry_discarded = False
        compact_input_geometry_isolated = False
        preliminary_input_bounds_hint: tuple[int, int, int, int] | None = None
        icon_cluster_audit_used = False
        icon_cluster_audit_candidate_count = 0
        icon_cluster_audit_reload_attested = False
        icon_cluster_localization_used = False
        icon_cluster_localization_roi_bounds: tuple[int, int, int, int] | None = None
        icon_cluster_local_geometry_verified = False
        icon_cluster_local_geometry_bounds: tuple[int, int, int, int] | None = None
        input_structure_audit_used = False
        input_structure_audit_retry_used = False
        input_structure_audit_isolated_from_attested_non_input = False
        input_lineage_used = False
        compact_reused_from_typed_lineage = False
        observation_cache_hit = False
        system_ui_audit_used = False
        system_ui_audit_retry_used = False
        system_ui_audit_confidence: float | None = None
        system_ui_audit_evidence: tuple[str, ...] = ()
        targeted_roi_bounds: tuple[int, int, int, int] | None = None
        icon_cluster_audit_roi_bounds: tuple[int, int, int, int] | None = None
        stable_tail_start = 0
        visual_obstructions: tuple[VisualObstruction, ...] = ()
        model_call_elapsed_seconds: list[float] = []
        model_call_token_budgets: list[int] = []

        def model_chat(messages: list[dict[str, Any]], *, max_tokens: int) -> str:
            nonlocal model_calls
            model_calls += 1
            model_call_token_budgets.append(max_tokens)
            call_started = time.perf_counter()
            try:
                scope_factory = getattr(self.provider, "call_scope", None)
                scope = (
                    scope_factory(
                        stage=self._current_stage,
                        fingerprint=fingerprint,
                    )
                    if callable(scope_factory)
                    else nullcontext()
                )
                with scope:
                    return self._provider_chat(messages, max_tokens=max_tokens)
            finally:
                model_identity.clear()
                model_identity.update(public_model_identity(self.provider.status()))
                model_call_elapsed_seconds.append(
                    round(time.perf_counter() - call_started, 3)
                )

        try:
            if len(frames) < 4:
                raise VisionAgentError("通用页面观察至少需要4帧。")
            stability = measure_local_stability(
                frames,
                allow_leading_outlier=True,
            )
            if not stability.stable:
                self.last_diagnostics = {
                    "observer_version": GENERIC_SCENE_OBSERVER_VERSION,
                    "vision_model": model_identity,
                    "model_calls": 0,
                    "local_stability": stability.to_dict(),
                    "failed_stage": "checking_stability",
                }
                raise VisionAgentError(
                    f"本地多帧稳定性检查未通过：{stability.reason}；不调用模型。"
                )

            sharpness_scores = [measure_frame_sharpness(item) for item in frames]
            # Stability intentionally permits one stale leading frame, so that
            # frame cannot be selected again merely because a transient vendor
            # overlay makes it look artificially sharp.  Only the converged
            # tail is eligible to become the trusted observation.
            stable_tail_start = max(0, len(frames) - min(3, len(frames)))
            selected_frame_index = max(
                range(stable_tail_start, len(frames)),
                key=sharpness_scores.__getitem__,
            )
            frame = frames[selected_frame_index].convert("RGB")
            visual_obstructions = consensus_top_edge_obstructions(
                frames[stable_tail_start:]
            )
            fingerprint = _local_frame_fingerprint(frame)
            context = _safe_goal_context(goal_context or {})
            cache_key = _observation_cache_key(
                device_id=device_id,
                fingerprint=fingerprint,
                goal_context=context,
                input_lineage=input_lineage_override,
            )
            cached_scene: UIScene | None = None
            if cache_key is not None:
                with self._observation_cache_lock:
                    cached_scene = self._observation_cache.get(cache_key)
                    if cached_scene is not None:
                        self._observation_cache.move_to_end(cache_key)
            if cached_scene is not None:
                observation_cache_hit = True
                cache_recorder = getattr(
                    self.provider,
                    "record_observation_cache_hit",
                    None,
                )
                if callable(cache_recorder):
                    cache_recorder(
                        stage="same_fingerprint_observation",
                        fingerprint=fingerprint,
                    )
                self.last_diagnostics = {
                    "observer_version": GENERIC_SCENE_OBSERVER_VERSION,
                    "vision_model": model_identity,
                    "strategy": "exact_fingerprint_context_cache",
                    "model_calls": 0,
                    "observation_cache_hit": True,
                    "fingerprint": fingerprint,
                    "element_count": len(cached_scene.elements),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
                self._set_stage("completed")
                return cached_scene
            allow_omitted_local_input_auxiliary_confirmation = bool(
                context.pop(
                    "_allow_omitted_local_input_auxiliary_confirmation",
                    False,
                )
                is True
            )
            privacy_minimized_system_home = (
                _goal_requests_coordinate_free_system_home(context)
            )
            model_frame = (
                privacy_minimized_system_navigation_view(frame)
                if privacy_minimized_system_home
                else frame
            )
            system_ui_audit_required = _goal_requests_system_ui_audit(context)
            camera_layout_orientation = _camera_layout_orientation(frame)
            image_part = {
                "type": "image_url",
                "image_url": {"url": _image_data_url(model_frame)},
            }
            detail_image_part = image_part
            first_messages = [
                _json_only_system_message(),
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _compact_prompt(context)},
                        image_part,
                    ],
                }
            ]

            continuation_base = _typed_input_continuation_base_scene(
                prior_scene=prior_scene,
                input_lineage=input_lineage_override,
                device_id=device_id,
                goal_context=context,
                fingerprint=fingerprint,
            )
            if continuation_base is not None:
                compact_reused_from_typed_lineage = True
                compact_input_geometry_isolated = True
                scene = continuation_base
                preliminary_input_bounds_hint = tuple(
                    round(value * 1000)
                    for value in input_lineage_override.input_bounds
                )

            def parse_compact_response(value: str) -> UIScene:
                nonlocal compact_geometry_discarded, compact_input_geometry_isolated
                nonlocal preliminary_input_bounds_hint
                payload = _extract_compact_json_object(value)
                preliminary_input_bounds_hint = _unique_payload_input_bounds(
                    payload
                )
                compact_input_geometry_isolated = (
                    _strip_preliminary_input_geometry_for_dedicated_audit(
                        payload,
                        context,
                    )
                )
                compact_input_geometry_isolated = (
                    _strip_preliminary_keyboard_containers_for_dedicated_audit(
                        payload,
                        context,
                    )
                    or compact_input_geometry_isolated
                )
                compact_geometry_discarded = (
                    _discard_compact_elements_for_targeted_geometry_recovery(
                        payload,
                        context,
                    )
                )
                return _suppress_obscured_input_evidence(
                    _parse_scene(
                        json.dumps(
                            payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        fingerprint=fingerprint,
                        goal_context=context,
                        allow_invalid_system_ui_unknown=True,
                        camera_layout_orientation=camera_layout_orientation,
                    ),
                    visual_obstructions,
                    fingerprint=fingerprint,
                )

            def parse_unique_structural_repair(value: str) -> UIScene | None:
                repaired = _parse_scene_after_unique_structural_edit(
                    value,
                    fingerprint=fingerprint,
                    goal_context=context,
                    allow_invalid_system_ui_unknown=True,
                    camera_layout_orientation=camera_layout_orientation,
                )
                if repaired is None:
                    return None
                return _suppress_obscured_input_evidence(
                    repaired,
                    visual_obstructions,
                    fingerprint=fingerprint,
                )

            if not compact_reused_from_typed_lineage:
                self._set_stage("waiting_compact_observation")
                try:
                    raw = model_chat(
                        first_messages,
                        max_tokens=COMPACT_OUTPUT_TOKENS,
                    )
                    self.last_raw_response = raw
                    self._set_stage("parsing_compact_observation")
                    scene = parse_compact_response(raw)
                except VisionAgentError as first_error:
                    first_error_type = classify_qwen_error(
                        first_error,
                        raw_response=self.last_raw_response,
                    )
                    if (
                        first_error_type not in FORMAT_ERROR_TYPES
                        or not _compact_response_has_repairable_syntax_error(
                            self.last_raw_response
                        )
                    ):
                        raise
                    scene = parse_unique_structural_repair(self.last_raw_response)
                    if scene is None:
                        raise VisionAgentError(
                            "原始 compact 响应不存在唯一、严格有效的单结构标点修复。"
                        )
                    format_retry_used = True
                    local_structural_repair_used = True

            if privacy_minimized_system_home:
                # The masked view is authority only for a coordinate-free
                # system Home choice.  App identity, page completion and any
                # element candidates from this view are deliberately erased.
                scene = replace(
                    scene,
                    app_id="unknown",
                    screen_id="unknown",
                    summary="中央App内容未披露；仅建立系统Home前稳定画布观察。",
                    elements=(),
                    overlays=(),
                )
                scene.validate()

            if (
                not compact_reused_from_typed_lineage
                and not system_ui_audit_required
                and not _goal_requests_keyboard_mode_switch(context)
                and not compact_input_geometry_isolated
                and not _is_verified_navigation_result_observation(context)
                and (
                    compact_geometry_discarded
                    or _needs_targeted_refinement(scene, context)
                )
            ):
                targeted_refinement_used = True
                targeted_roi_bounds = _goal_directed_roi_bounds(context)
                self._set_stage("waiting_targeted_refinement")
                detail_messages = [
                    _json_only_system_message(),
                    {
                        "role": "user",
                        "content": (
                            [
                                {
                                "type": "text",
                                "text": _targeted_prompt(
                                    context,
                                    first_scene=scene.to_dict(),
                                    roi_bounds=None,
                                ),
                                },
                                image_part,
                            ]
                        ),
                    }
                ]
                try:
                    raw = model_chat(
                        detail_messages,
                        max_tokens=TARGETED_OUTPUT_TOKENS,
                    )
                    self.last_raw_response = raw
                    self._set_stage("parsing_targeted_refinement")
                    # A failed refinement must stop the controller. Returning the
                    # earlier ambiguous scene would allow action on stale evidence.
                    scene = _suppress_obscured_input_evidence(
                        _parse_targeted_scene_delta(
                            raw,
                            base_scene=scene,
                            fingerprint=fingerprint,
                            goal_context=context,
                        ),
                        visual_obstructions,
                        fingerprint=fingerprint,
                    )

                except VisionAgentError as targeted_error:
                    targeted_error_type = classify_qwen_error(
                        targeted_error,
                        raw_response=self.last_raw_response,
                    )
                    if (
                        format_retry_used
                        or targeted_error_type not in FORMAT_ERROR_TYPES
                        or not _targeted_response_has_repairable_syntax_error(
                            self.last_raw_response
                        )
                    ):
                        raise
                    scene = _parse_targeted_delta_after_unique_structural_edit(
                        self.last_raw_response,
                        base_scene=scene,
                        fingerprint=fingerprint,
                        goal_context=context,
                    )
                    if scene is None:
                        raise VisionAgentError(
                            "目标精查响应不存在唯一、严格有效的单结构标点修复。"
                        ) from targeted_error
                    scene = _suppress_obscured_input_evidence(
                        scene,
                        visual_obstructions,
                        fingerprint=fingerprint,
                    )
                    format_retry_used = True
                    local_structural_repair_used = True

            if (
                not compact_reused_from_typed_lineage
                and not privacy_minimized_system_home
                and _needs_foreground_app_identity_audit(scene, context)
            ):
                foreground_app_identity_audit_used = True
                self._set_stage("waiting_foreground_app_identity_audit")
                raw = model_chat(
                    [
                        _json_only_system_message(),
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": _foreground_app_identity_audit_prompt(),
                                },
                                image_part,
                            ],
                        },
                    ],
                    max_tokens=FOREGROUND_APP_IDENTITY_AUDIT_TOKENS,
                )
                self.last_raw_response = raw
                self._set_stage("parsing_foreground_app_identity_audit")
                (
                    foreground_app_id,
                    foreground_app_identity_audit_confidence,
                    foreground_app_identity_audit_evidence,
                ) = _strict_foreground_app_identity_audit(raw)
                scene = replace(scene, app_id=foreground_app_id)
                scene.validate()

            if (
                not compact_reused_from_typed_lineage
                and _goal_requests_reload(context)
            ):
                # Reload is a generic navigation semantic, but compact toolbar
                # glyphs are easy to confuse with bookmark and expand controls.
                # A separate read-only audit is the only component allowed to
                # mint the local reload_visual_audit fact used by the policy.
                icon_cluster_audit_used = True
                icon_cluster_audit_roi_bounds = (
                    targeted_roi_bounds or _goal_directed_roi_bounds(context)
                )
                self._set_stage("waiting_icon_cluster_audit")
                icon_audit_content: list[dict[str, Any]] = [
                    {
                        "type": "text",
                        "text": _icon_cluster_audit_prompt(
                            context,
                            roi_bounds=None,
                        ),
                    },
                    image_part,
                ]
                raw = model_chat(
                    [
                        _json_only_system_message(),
                        {"role": "user", "content": icon_audit_content},
                    ],
                    max_tokens=ICON_CLUSTER_AUDIT_TOKENS,
                )
                self.last_raw_response = raw
                self._set_stage("parsing_icon_cluster_audit")
                rough_audit = _strict_icon_cluster_audit_payload(raw)
                icon_cluster_localization_roi_bounds = (
                    _attestable_icon_cluster_bounds(
                        rough_audit,
                        roi_bounds=icon_cluster_audit_roi_bounds,
                    )
                )
                final_audit_raw = raw
                if icon_cluster_localization_roi_bounds is not None:
                    icon_cluster_localization_used = True
                    icon_cluster_frame = _crop_normalized(
                        frame,
                        icon_cluster_localization_roi_bounds,
                    )
                    self._set_stage("waiting_icon_cluster_localization")
                    final_audit_raw = model_chat(
                        [
                            _json_only_system_message(),
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": _icon_cluster_localization_prompt(),
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": _image_data_url(icon_cluster_frame)
                                        },
                                    },
                                ],
                            },
                        ],
                        max_tokens=ICON_CLUSTER_AUDIT_TOKENS,
                    )
                    self.last_raw_response = final_audit_raw
                    self._set_stage("parsing_icon_cluster_localization")
                    localized_audit = _strict_icon_cluster_audit_payload(
                        final_audit_raw
                    )
                    mapped_audit = _map_icon_cluster_audit_to_full_frame(
                        localized_audit,
                        icon_cluster_localization_roi_bounds,
                    )
                    snapped_audit, icon_cluster_local_geometry_bounds = (
                        _snap_reload_audit_to_local_glyph(
                            frame,
                            mapped_audit,
                            search_bounds=icon_cluster_localization_roi_bounds,
                        )
                    )
                    icon_cluster_local_geometry_verified = (
                        snapped_audit is not None
                    )
                    final_audit_raw = json.dumps(
                        snapped_audit or mapped_audit,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                (
                    scene,
                    icon_cluster_audit_candidate_count,
                    icon_cluster_audit_reload_attested,
                ) = _apply_icon_cluster_audit(
                    scene,
                    final_audit_raw,
                    fingerprint=fingerprint,
                    localization_verified=(
                        icon_cluster_localization_used
                        and icon_cluster_local_geometry_verified
                    ),
                    roi_bounds=(
                        icon_cluster_localization_roi_bounds
                        or icon_cluster_audit_roi_bounds
                    ),
                )

            if not compact_reused_from_typed_lineage and system_ui_audit_required:
                system_ui_audit_used = True
                system_ui_images = [
                    image_part,
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(
                            frame.transpose(Image.Transpose.ROTATE_90)
                        )},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(
                            frame.transpose(Image.Transpose.ROTATE_270)
                        )},
                    },
                ]
                self._set_stage("waiting_system_ui_audit")
                audit_messages = [
                    _json_only_system_message(),
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": _system_ui_audit_prompt(context),
                            },
                            *system_ui_images,
                        ],
                    },
                ]
                raw = model_chat(
                    audit_messages,
                    max_tokens=SYSTEM_UI_AUDIT_TOKENS,
                )
                self.last_raw_response = raw
                self._set_stage("parsing_system_ui_audit")
                scene, system_ui_audit_confidence, system_ui_audit_evidence = (
                    _apply_system_ui_audit(
                        scene,
                        raw,
                        fingerprint=fingerprint,
                    )
                )

            if _should_audit_prefilled_input(scene, context):
                input_structure_audit_used = True
                input_audit_base_scene = scene
                active_input_field_id = _goal_active_input_field(context)[0]
                verified_input_lineage: TypedInputLineage | None = None
                if (
                    input_lineage_override is not None
                    and isinstance(device_id, str)
                    and input_lineage_override.matches_typed_context(
                        device_id=device_id,
                        app_id=scene.app_id,
                        screen_id=scene.screen_id,
                        input_field_id=active_input_field_id,
                    )
                ):
                    verified_input_lineage = input_lineage_override
                if (
                    verified_input_lineage is None
                    and self.input_lineage_store is not None
                    and isinstance(device_id, str)
                    and device_id.strip()
                ):
                    stored_input_lineage = self.input_lineage_store.load(device_id)
                    if (
                        stored_input_lineage is not None
                        and stored_input_lineage.matches_typed_context(
                            device_id=device_id,
                            app_id=scene.app_id,
                            screen_id=scene.screen_id,
                            input_field_id=active_input_field_id,
                            now_epoch=float(self.input_lineage_store.clock()),
                            ttl_seconds=self.input_lineage_store.ttl_seconds,
                        )
                    ):
                        verified_input_lineage = stored_input_lineage
                if verified_input_lineage is not None:
                    input_lineage_used = True
                # The compact pass may describe the page, but it is not an
                # input-value authority.  Only a typed lineage may seed the
                # dedicated audit's deterministic next-key whitelist; the
                # final committed/preedit values are reduced from that audit.
                input_ledger_value_hint = (
                    verified_input_lineage.exact_value
                    if verified_input_lineage is not None
                    else None
                )
                self._set_stage("waiting_input_structure_audit")
                temporal_input_frames = (
                    tuple(frames[stable_tail_start:])
                    if (
                        _goal_requests_active_verified_text_clear(context)
                        or _goal_active_input_field(context)[2]
                    )
                    else (frame,)
                )
                if not temporal_input_frames:
                    temporal_input_frames = (frame,)
                input_audit_prompt = _input_structure_audit_prompt(
                    context,
                    roi_bounds=None,
                    current_input_text=input_ledger_value_hint,
                )
                if len(temporal_input_frames) > 1:
                    input_audit_prompt += (
                        "\nImages 1.."
                        f"{len(temporal_input_frames)} are aligned captures of the "
                        "same stable phone surface at different times. A text caret "
                        "may blink, so derive caret_line_index from any frame where "
                        "the complete caret is visible. All normalized geometry uses "
                        "the same 0..1000 frame and must remain consistent across the "
                        "images; do not merge any other transient content."
                    )
                audit_content: list[dict[str, Any]] = [
                    {
                        "type": "text",
                        "text": input_audit_prompt,
                    },
                    *(
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_data_url(item)},
                        }
                        for item in temporal_input_frames
                    ),
                ]
                raw = model_chat(
                    [
                        _json_only_system_message(),
                        {"role": "user", "content": audit_content},
                    ],
                    max_tokens=INPUT_STRUCTURE_AUDIT_TOKENS,
                )
                self.last_raw_response = raw
                self._set_stage("parsing_input_structure_audit")
                try:
                    scene = _suppress_obscured_input_evidence(
                        _apply_input_structure_audit(
                            scene,
                            raw,
                            fingerprint=fingerprint,
                            goal_context=context,
                            ledger_input_value=input_ledger_value_hint,
                            verified_input_lineage=verified_input_lineage,
                            device_id=device_id,
                            lineage_frame=frame,
                            qwerty_row_snapper=self.qwerty_row_snapper,
                            qwerty_row_frames=frames[stable_tail_start:],
                        ),
                        visual_obstructions,
                        fingerprint=fingerprint,
                    )
                    input_retry_roi = _input_audit_retry_roi(
                        context,
                        preliminary_input_bounds_hint=preliminary_input_bounds_hint,
                        first_audit_raw=raw,
                    )
                    if (
                        _goal_has_explicit_input_text(context)
                        and input_retry_roi is not None
                        and not _input_audit_established_local_target(scene)
                    ):
                        # A valid empty audit grants no geometry authority. One
                        # independent single-crop retry is allowed from either
                        # a goal-authored coarse region or one uniquely isolated
                        # compact input box. The hint grants crop selection only;
                        # the crop owns a local 0..1000 coordinate system and is
                        # mapped back locally. The first empty result contributes
                        # no fields, bounds or states. Two empty results fail as
                        # an observation fault below.
                        input_structure_audit_retry_used = True
                        self._set_stage("waiting_input_structure_audit")
                        retry_content = [
                            {
                                "type": "text",
                                "text": _input_structure_audit_prompt(
                                    context,
                                    roi_bounds=input_retry_roi,
                                    crop_local=True,
                                    current_input_text=input_ledger_value_hint,
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": _image_data_url(
                                        _crop_normalized(frame, input_retry_roi)
                                    )
                                },
                            },
                        ]
                        raw = model_chat(
                            [
                                _json_only_system_message(),
                                {"role": "user", "content": retry_content},
                            ],
                            max_tokens=INPUT_STRUCTURE_AUDIT_TOKENS,
                        )
                        self.last_raw_response = raw
                        self._set_stage("parsing_input_structure_audit")
                        raw = _map_input_structure_crop_audit_to_full(
                            raw,
                            roi_bounds=input_retry_roi,
                        )
                        scene = _suppress_obscured_input_evidence(
                            _apply_input_structure_audit(
                                input_audit_base_scene,
                                raw,
                                fingerprint=fingerprint,
                                goal_context=context,
                                ledger_input_value=input_ledger_value_hint,
                                verified_input_lineage=verified_input_lineage,
                                device_id=device_id,
                                lineage_frame=frame,
                                qwerty_row_snapper=self.qwerty_row_snapper,
                                qwerty_row_frames=frames[stable_tail_start:],
                            ),
                            visual_obstructions,
                            fingerprint=fingerprint,
                        )
                    if (
                        _goal_active_input_transaction_text(context)
                        and not _input_audit_established_local_target(scene)
                        and not allow_omitted_local_input_auxiliary_confirmation
                    ):
                        raise VisionAgentError(
                            "专用输入结构审计没有建立当前输入事务的唯一本地目标。"
                        )
                except VisionAgentError as audit_error:
                    try:
                        scene = _apply_hidden_keyboard_only_attestation(
                            input_audit_base_scene,
                            raw,
                            fingerprint=fingerprint,
                            goal_context=context,
                        )
                    except VisionAgentError:
                        if not _can_isolate_input_audit_from_attested_non_input(
                            scene,
                            context,
                        ):
                            raise audit_error
                        # The rejected input payload contributes no fields or
                        # geometry. A separately localized reload glyph remains a
                        # valid non-input target even when the broader natural
                        # language goal also mentions the post-reload field state.
                        input_structure_audit_isolated_from_attested_non_input = True

            missing_goal_evidence = [
                element.element_id
                for element in scene.elements
                if element.states.get("goal_relevant") is True
                and not any(item.strip() for item in element.evidence)
            ]
            if missing_goal_evidence:
                self.last_diagnostics = {
                    "observer_version": GENERIC_SCENE_OBSERVER_VERSION,
                    "vision_model": model_identity,
                    "strategy": "compact_then_targeted_on_demand",
                    "model_calls": model_calls,
                    "targeted_refinement_used": targeted_refinement_used,
                    "compact_geometry_discarded": compact_geometry_discarded,
                    "compact_input_geometry_isolated": (
                        compact_input_geometry_isolated
                    ),
                    "missing_goal_evidence_element_ids": missing_goal_evidence,
                    "fingerprint": fingerprint,
                }
                raise VisionAgentError(
                    "目标相关元素缺少原始可见证据，不能建立可信候选："
                    + ",".join(missing_goal_evidence)
                )

            target_local_candidate = scene.unique_trusted_goal_element()
            completion_evidence = scene.trusted_completion_evidence()
            if not scene.stable or (
                float(scene.confidence) < MIN_TARGET_CONFIDENCE
                and target_local_candidate is None
                and not completion_evidence
            ):
                self.last_diagnostics = {
                    "observer_version": GENERIC_SCENE_OBSERVER_VERSION,
                    "vision_model": model_identity,
                    "strategy": "compact_then_targeted_on_demand",
                    "model_calls": model_calls,
                    "compact_retry_used": compact_retry_used,
                    "format_retry_used": format_retry_used,
                    "local_structural_repair_used": local_structural_repair_used,
                    "targeted_refinement_used": targeted_refinement_used,
                    "compact_geometry_discarded": compact_geometry_discarded,
                    "icon_cluster_audit_used": icon_cluster_audit_used,
                    "icon_cluster_audit_candidate_count": (
                        icon_cluster_audit_candidate_count
                    ),
                    "icon_cluster_audit_reload_attested": (
                        icon_cluster_audit_reload_attested
                    ),
                    "icon_cluster_localization_used": (
                        icon_cluster_localization_used
                    ),
                    "icon_cluster_local_geometry_verified": (
                        icon_cluster_local_geometry_verified
                    ),
                    "local_stability": stability.to_dict(),
                    "selected_frame_index": selected_frame_index,
                    "stable_tail_start_index": stable_tail_start,
                    "visual_obstructions": [
                        item.to_dict() for item in visual_obstructions
                    ],
                    "scene_confidence": float(scene.confidence),
                    "candidate_summary": [
                        {
                            "element_id": item.element_id,
                            "role": item.role,
                            "meaning": item.meaning,
                            "label": item.label,
                            "confidence": float(item.confidence),
                            "goal_relevant": item.states.get("goal_relevant") is True,
                        }
                        for item in scene.elements
                    ],
                }
                raise VisionAgentError(
                    "页面不稳定或整体置信度不足，不能建立可信候选。"
                )

            self.last_diagnostics = {
                "observer_version": GENERIC_SCENE_OBSERVER_VERSION,
                "vision_model": model_identity,
                "strategy": "compact_then_targeted_on_demand",
                "model_calls": model_calls,
                "compact_retry_used": compact_retry_used,
                "format_retry_used": format_retry_used,
                "local_structural_repair_used": local_structural_repair_used,
                "first_pass_success": not format_retry_used,
                "repair_retry_success": format_retry_used,
                "targeted_refinement_used": targeted_refinement_used,
                "foreground_app_identity_audit_used": (
                    foreground_app_identity_audit_used
                ),
                "foreground_app_identity_audit_confidence": (
                    foreground_app_identity_audit_confidence
                ),
                "foreground_app_identity_audit_evidence": list(
                    foreground_app_identity_audit_evidence
                ),
                "system_navigation_privacy_view_version": (
                    SYSTEM_NAVIGATION_PRIVACY_VIEW_VERSION
                    if privacy_minimized_system_home
                    else None
                ),
                "compact_geometry_discarded": compact_geometry_discarded,
                "compact_input_geometry_isolated": compact_input_geometry_isolated,
                "icon_cluster_audit_used": icon_cluster_audit_used,
                "icon_cluster_audit_candidate_count": (
                    icon_cluster_audit_candidate_count
                ),
                "icon_cluster_audit_reload_attested": (
                    icon_cluster_audit_reload_attested
                ),
                "icon_cluster_localization_used": icon_cluster_localization_used,
                "icon_cluster_local_geometry_verified": (
                    icon_cluster_local_geometry_verified
                ),
                "icon_cluster_local_geometry_bounds": (
                    list(icon_cluster_local_geometry_bounds)
                    if icon_cluster_local_geometry_bounds is not None
                    else None
                ),
                "icon_cluster_localization_roi_bounds": (
                    list(icon_cluster_localization_roi_bounds)
                    if icon_cluster_localization_roi_bounds is not None
                    else None
                ),
                "icon_cluster_audit_roi_bounds": (
                    list(icon_cluster_audit_roi_bounds)
                    if icon_cluster_audit_roi_bounds is not None
                    else None
                ),
                "input_structure_audit_used": input_structure_audit_used,
                "input_lineage_used": input_lineage_used,
                "compact_reused_from_typed_lineage": (
                    compact_reused_from_typed_lineage
                ),
                "observation_cache_hit": observation_cache_hit,
                "input_structure_audit_retry_used": (
                    input_structure_audit_retry_used
                ),
                "input_structure_audit_isolated_from_attested_non_input": (
                    input_structure_audit_isolated_from_attested_non_input
                ),
                "system_ui_audit_used": system_ui_audit_used,
                "system_ui_audit_retry_used": system_ui_audit_retry_used,
                "system_ui_audit_confidence": system_ui_audit_confidence,
                "system_ui_audit_evidence": list(system_ui_audit_evidence),
                "targeted_roi_bounds": (
                    list(targeted_roi_bounds)
                    if targeted_roi_bounds is not None
                    else None
                ),
                "prefilled_input_structure_inferred": any(
                    item.element_id.startswith("local_structured_input_")
                    for item in scene.elements
                ),
                "model_call_elapsed_seconds": model_call_elapsed_seconds,
                "model_call_token_budgets": model_call_token_budgets,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "local_stability": stability.to_dict(),
                "selected_frame_index": selected_frame_index,
                "stable_tail_start_index": stable_tail_start,
                "visual_obstructions": [
                    item.to_dict() for item in visual_obstructions
                ],
                "confidence_basis": (
                    "scene"
                    if float(scene.confidence) >= MIN_TARGET_CONFIDENCE
                    else "unique_goal_element"
                    if target_local_candidate is not None
                    else "completion_evidence_only"
                ),
                "frame_sharpness_scores": [
                    round(value, 3) for value in sharpness_scores
                ],
                "frame_size": list(frame.size),
                "fingerprint": fingerprint,
                "element_count": len(scene.elements),
                "output_token_budget": (
                    model_call_token_budgets[-1]
                    if model_call_token_budgets
                    else 0
                ),
            }
            if cache_key is not None:
                with self._observation_cache_lock:
                    self._observation_cache[cache_key] = scene
                    self._observation_cache.move_to_end(cache_key)
                    while len(self._observation_cache) > self._observation_cache_limit:
                        self._observation_cache.popitem(last=False)
            self._set_stage("completed")
            return scene
        except Exception as exc:
            failed_stage = self.status()["last_stage"]
            self._set_stage("failed")
            base = dict(self.last_diagnostics)
            base.update(
                {
                    "observer_version": GENERIC_SCENE_OBSERVER_VERSION,
                    "vision_model": model_identity,
                    "model_calls": model_calls,
                    "compact_retry_used": compact_retry_used,
                    "format_retry_used": format_retry_used,
                    "local_structural_repair_used": local_structural_repair_used,
                    "first_pass_success": False,
                    "repair_retry_success": False,
                    "targeted_refinement_used": targeted_refinement_used,
                    "compact_geometry_discarded": compact_geometry_discarded,
                    "icon_cluster_audit_used": icon_cluster_audit_used,
                    "icon_cluster_audit_candidate_count": (
                        icon_cluster_audit_candidate_count
                    ),
                    "icon_cluster_audit_reload_attested": (
                        icon_cluster_audit_reload_attested
                    ),
                    "input_structure_audit_used": input_structure_audit_used,
                    "compact_reused_from_typed_lineage": (
                        compact_reused_from_typed_lineage
                    ),
                    "observation_cache_hit": observation_cache_hit,
                    "input_structure_audit_retry_used": (
                        input_structure_audit_retry_used
                    ),
                    "input_structure_audit_isolated_from_attested_non_input": (
                        input_structure_audit_isolated_from_attested_non_input
                    ),
                    "system_ui_audit_used": system_ui_audit_used,
                    "system_ui_audit_retry_used": system_ui_audit_retry_used,
                    "stable_tail_start_index": stable_tail_start,
                    "visual_obstructions": [
                        item.to_dict() for item in visual_obstructions
                    ],
                    "model_call_elapsed_seconds": model_call_elapsed_seconds,
                    "model_call_token_budgets": model_call_token_budgets,
                }
            )
            base.update(
                failure_diagnostics(
                    exc,
                    raw_response=self.last_raw_response,
                    stage=failed_stage,
                    model_calls=model_calls,
                    elapsed_seconds=time.perf_counter() - started,
                    safe_stop_reason="观察阶段未建立可信候选，决策模型与控制器均未执行动作。",
                )
            )
            base["raw_response_length"] = len(self.last_raw_response)
            base["raw_response_excerpt"] = self.last_raw_response[:1000]
            base["scene_enum_values"] = _scene_enum_values(
                self.last_raw_response
            )
            if input_structure_audit_used:
                base["input_structure_shape"] = _input_structure_diagnostic_shape(
                    self.last_raw_response
                )
            self.last_diagnostics = base
            raise
        finally:
            self._set_stage("idle")

    def _provider_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        response_format: dict[str, str] | None = None,
    ) -> str:
        try:
            options: dict[str, Any] = {
                "timeout": OBSERVATION_TIMEOUT_SECONDS,
                # One closed-loop observation owns one network attempt.  A
                # malformed response is rejected locally and can never spend
                # a second Qwen request inside the same step.
                "max_attempts": 1,
            }
            options["response_format"] = response_format or {"type": "json_object"}
            return self.provider._chat(
                messages,
                max_tokens=max_tokens,
                **options,
            )
        except TypeError as exc:
            # Keep simple test providers and local replay providers compatible.
            # Production DashScopeVisionProvider accepts the explicit limits.
            text = str(exc)
            if "unexpected keyword" not in text and "keyword argument" not in text:
                raise
            fallback_options = dict(options)
            fallback_options.pop("response_format", None)
            try:
                return self.provider._chat(
                    messages,
                    max_tokens=max_tokens,
                    **fallback_options,
                )
            except TypeError as fallback_exc:
                fallback_text = str(fallback_exc)
                if (
                    "unexpected keyword" not in fallback_text
                    and "keyword argument" not in fallback_text
                ):
                    raise
                return self.provider._chat(messages, max_tokens=max_tokens)


class SingleStepGenericSceneObserver(GenericSceneObserver):
    """Production observer backed by at most one Qwen request per fresh scene.

    The response carries the ordinary scene and, only for an input-related
    subgoal, the complete input/IME/keyboard structure in one envelope.  All
    former targeted, App-identity, system-UI and input follow-up audits remain
    available above for isolated historical replay, but this production class
    never enters them.
    """

    def status(self) -> dict[str, Any]:
        value = super().status()
        value.update(
            {
                "observer_version": SINGLE_STEP_SCENE_OBSERVER_VERSION,
                "single_step_observation_protocol": (
                    SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION
                ),
                "model_role": "single_step_fused_observation",
                "max_online_calls_per_observation": 1,
                "single_step_output_tokens": SINGLE_STEP_OUTPUT_TOKENS,
            }
        )
        return value

    def observe(
        self,
        *,
        frames: list[Image.Image],
        goal_context: dict[str, Any] | None = None,
        device_id: str | None = None,
        input_lineage_override: TypedInputLineage | None = None,
        prior_scene: UIScene | None = None,
    ) -> UIScene:
        del prior_scene  # Fresh pixels and typed lineage are the only inputs.
        self.last_raw_response = ""
        model_identity = public_model_identity(self.provider.status())
        self.last_diagnostics = {"vision_model": model_identity}
        self._set_stage("checking_stability")
        started = time.perf_counter()
        model_calls = 0
        fingerprint = ""
        selected_frame_index = 0
        stable_tail_start = 0
        input_structure_required = False
        try:
            if len(frames) < 4:
                raise VisionAgentError("通用页面观察至少需要4帧。")
            stability = measure_local_stability(
                frames,
                allow_leading_outlier=True,
            )
            if not stability.stable:
                raise VisionAgentError(
                    f"本地多帧稳定性检查未通过：{stability.reason}；不调用模型。"
                )

            sharpness_scores = [measure_frame_sharpness(item) for item in frames]
            stable_tail_start = max(0, len(frames) - min(3, len(frames)))
            selected_frame_index = max(
                range(stable_tail_start, len(frames)),
                key=sharpness_scores.__getitem__,
            )
            frame = frames[selected_frame_index].convert("RGB")
            fingerprint = _local_frame_fingerprint(frame)
            context = _safe_goal_context(goal_context or {})
            cache_key = _observation_cache_key(
                device_id=device_id,
                fingerprint=fingerprint,
                goal_context=context,
                input_lineage=input_lineage_override,
            )
            if cache_key is not None:
                with self._observation_cache_lock:
                    cached = self._observation_cache.get(cache_key)
                    if cached is not None:
                        self._observation_cache.move_to_end(cache_key)
                if cached is not None:
                    recorder = getattr(
                        self.provider,
                        "record_observation_cache_hit",
                        None,
                    )
                    if callable(recorder):
                        recorder(
                            stage="same_fingerprint_single_step_observation",
                            fingerprint=fingerprint,
                        )
                    self.last_diagnostics = {
                        "observer_version": SINGLE_STEP_SCENE_OBSERVER_VERSION,
                        "vision_model": model_identity,
                        "strategy": "single_step_exact_fingerprint_cache",
                        "model_calls": 0,
                        "observation_cache_hit": True,
                        "fingerprint": fingerprint,
                        "element_count": len(cached.elements),
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                    }
                    self._set_stage("completed")
                    return cached

            input_structure_required = _goal_requests_input(context)
            active_field_id = _goal_active_input_field(context)[0]
            verified_lineage: TypedInputLineage | None = None
            if input_lineage_override is not None:
                verified_lineage = input_lineage_override
            ledger_value_hint = (
                verified_lineage.exact_value
                if verified_lineage is not None
                else None
            )
            privacy_minimized_system_home = (
                _goal_requests_coordinate_free_system_home(context)
            )
            model_frames = (
                tuple(frames[stable_tail_start:])
                if input_structure_required
                else (frame,)
            )
            if privacy_minimized_system_home:
                model_frames = tuple(
                    privacy_minimized_system_navigation_view(item.convert("RGB"))
                    for item in model_frames
                )

            prompt = _single_step_observation_prompt(
                context,
                include_input_structure=input_structure_required,
                current_input_text=ledger_value_hint,
                image_count=len(model_frames),
            )
            content: list[dict[str, Any]] = [
                {"type": "text", "text": prompt}
            ]
            for index, item in enumerate(model_frames, start=1):
                content.extend(
                    (
                        {
                            "type": "text",
                            "text": f"IMAGE {index} - SAME STABLE PHONE SURFACE",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": _image_data_url(item.convert("RGB"))
                            },
                        },
                    )
                )
            self._set_stage("waiting_single_step_observation")
            scope_factory = getattr(self.provider, "call_scope", None)
            scope = (
                scope_factory(
                    stage="single_step_observation",
                    fingerprint=fingerprint,
                )
                if callable(scope_factory)
                else nullcontext()
            )
            call_started = time.perf_counter()
            model_calls = 1
            with scope:
                raw = self._provider_chat(
                    [
                        _json_only_system_message(),
                        {"role": "user", "content": content},
                    ],
                    max_tokens=SINGLE_STEP_OUTPUT_TOKENS,
                    response_format={"type": "json_object"},
                )
            call_elapsed = round(time.perf_counter() - call_started, 3)
            self.last_raw_response = raw
            self._set_stage("parsing_single_step_observation")
            envelope = _parse_single_step_observation_envelope(
                raw,
                input_structure_required=input_structure_required,
            )
            scene_payload = dict(envelope["scene"])
            if input_structure_required:
                _strip_preliminary_input_geometry_for_dedicated_audit(
                    scene_payload,
                    context,
                )
                _strip_preliminary_keyboard_containers_for_dedicated_audit(
                    scene_payload,
                    context,
                )
            obstructions = consensus_top_edge_obstructions(
                frames[stable_tail_start:]
            )
            scene = _suppress_obscured_input_evidence(
                _parse_scene(
                    json.dumps(
                        scene_payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    fingerprint=fingerprint,
                    goal_context=context,
                    allow_invalid_system_ui_unknown=True,
                    camera_layout_orientation=_camera_layout_orientation(frame),
                ),
                obstructions,
                fingerprint=fingerprint,
            )

            if privacy_minimized_system_home:
                scene = replace(
                    scene,
                    app_id="unknown",
                    screen_id="unknown",
                    summary="中央App内容未披露；仅建立系统Home前稳定画布观察。",
                    elements=(),
                    overlays=(),
                )
                scene.validate()

            if input_structure_required:
                # The lineage can be trusted only after the same response has
                # established the actual foreground App and screen identity.
                if verified_lineage is not None and not (
                    isinstance(device_id, str)
                    and verified_lineage.matches_typed_context(
                        device_id=device_id,
                        app_id=scene.app_id,
                        screen_id=scene.screen_id,
                        input_field_id=active_field_id,
                    )
                ):
                    verified_lineage = None
                if (
                    verified_lineage is None
                    and self.input_lineage_store is not None
                    and isinstance(device_id, str)
                    and device_id.strip()
                ):
                    stored = self.input_lineage_store.load(device_id)
                    if (
                        stored is not None
                        and stored.matches_typed_context(
                            device_id=device_id,
                            app_id=scene.app_id,
                            screen_id=scene.screen_id,
                            input_field_id=active_field_id,
                            now_epoch=float(self.input_lineage_store.clock()),
                            ttl_seconds=self.input_lineage_store.ttl_seconds,
                        )
                    ):
                        verified_lineage = stored
                ledger_value_hint = (
                    verified_lineage.exact_value
                    if verified_lineage is not None
                    else None
                )
                input_payload = envelope["input_structure"]
                assert isinstance(input_payload, dict)
                scene = _suppress_obscured_input_evidence(
                    _apply_input_structure_audit(
                        scene,
                        json.dumps(
                            input_payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        fingerprint=fingerprint,
                        goal_context=context,
                        ledger_input_value=ledger_value_hint,
                        verified_input_lineage=verified_lineage,
                        device_id=device_id,
                        lineage_frame=frame,
                        qwerty_row_snapper=self.qwerty_row_snapper,
                        qwerty_row_frames=frames[stable_tail_start:],
                    ),
                    obstructions,
                    fingerprint=fingerprint,
                )
                if (
                    _goal_active_input_transaction_text(context)
                    and not _input_audit_established_local_target(scene)
                ):
                    raise VisionAgentError(
                        "单步完整观察没有建立当前输入事务的唯一本地目标。"
                    )

            missing_evidence = [
                item.element_id
                for item in scene.elements
                if item.states.get("goal_relevant") is True
                and not any(value.strip() for value in item.evidence)
            ]
            if missing_evidence:
                raise VisionAgentError(
                    "目标相关元素缺少原始可见证据，不能建立可信候选："
                    + ",".join(missing_evidence)
                )
            target = scene.unique_trusted_goal_element()
            completion = scene.trusted_completion_evidence()
            if not scene.stable or (
                float(scene.confidence) < MIN_TARGET_CONFIDENCE
                and target is None
                and not completion
            ):
                raise VisionAgentError(
                    "页面不稳定或整体置信度不足，不能建立可信候选。"
                )

            self.last_diagnostics = {
                "observer_version": SINGLE_STEP_SCENE_OBSERVER_VERSION,
                "vision_model": public_model_identity(self.provider.status()),
                "strategy": "single_step_fused_observation",
                "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                "model_calls": 1,
                "online_stages": ["single_step_observation"],
                "input_structure_in_same_response": input_structure_required,
                "remote_retry_used": False,
                "observation_cache_hit": False,
                "selected_frame_index": selected_frame_index,
                "stable_tail_start_index": stable_tail_start,
                "local_stability": stability.to_dict(),
                "frame_sharpness_scores": [
                    round(value, 3) for value in sharpness_scores
                ],
                "frame_size": list(frame.size),
                "fingerprint": fingerprint,
                "element_count": len(scene.elements),
                "model_call_elapsed_seconds": [call_elapsed],
                "model_call_token_budgets": [SINGLE_STEP_OUTPUT_TOKENS],
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
            if cache_key is not None:
                with self._observation_cache_lock:
                    self._observation_cache[cache_key] = scene
                    self._observation_cache.move_to_end(cache_key)
                    while len(self._observation_cache) > self._observation_cache_limit:
                        self._observation_cache.popitem(last=False)
            self._set_stage("completed")
            return scene
        except Exception as exc:
            failed_stage = self.status()["last_stage"]
            self._set_stage("failed")
            self.last_diagnostics = {
                "observer_version": SINGLE_STEP_SCENE_OBSERVER_VERSION,
                "vision_model": public_model_identity(self.provider.status()),
                "strategy": "single_step_fused_observation",
                "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                "model_calls": model_calls,
                "online_stages": (
                    ["single_step_observation"] if model_calls else []
                ),
                "input_structure_in_same_response": input_structure_required,
                "remote_retry_used": False,
                "failed_stage": failed_stage,
                "fingerprint": fingerprint,
                "error": str(exc),
                "error_type": classify_qwen_error(
                    exc,
                    raw_response=self.last_raw_response,
                ),
                "safe_stop_reason": (
                    "单次模型输出未建立完整可信观察；没有发起第二次Qwen请求，"
                    "控制器与机械臂均未执行。"
                ),
                "raw_response_length": len(self.last_raw_response),
                "raw_response_excerpt": self.last_raw_response[:1000],
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
            raise
        finally:
            self._set_stage("idle")


def _horizontal_overlap_ratio(
    first: tuple[float, float, float, float],
    second: tuple[int, int, int, int],
) -> float:
    overlap = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    return overlap / max(1.0, first[2] - first[0])


def _input_evidence_is_obscured(
    bounds: tuple[float, float, float, float],
    obstruction: VisualObstruction,
) -> bool:
    if obstruction.kind != "top_edge_opaque_band":
        return False
    horizontal_overlap = _horizontal_overlap_ratio(bounds, obstruction.bounds)
    vertical_gap = bounds[1] - obstruction.bounds[3]
    obstruction_height = obstruction.bounds[3] - obstruction.bounds[1]
    return horizontal_overlap >= 0.15 and vertical_gap <= max(
        20.0,
        obstruction_height * 0.5,
    )


def _suppress_obscured_input_evidence(
    scene: UIScene,
    obstructions: tuple[VisualObstruction, ...],
    *,
    fingerprint: str,
) -> UIScene:
    """Prevent a crop/model claim from restoring pixels hidden in the full frame."""

    if not obstructions:
        return scene
    value = scene.to_dict()
    changed = False
    for element in value.get("elements") or []:
        if not isinstance(element, dict) or element.get("role") != "input":
            continue
        raw_bounds = element.get("bounds")
        if not isinstance(raw_bounds, list) or len(raw_bounds) != 4:
            continue
        bounds = tuple(float(part) * 1000 for part in raw_bounds)
        matched = next(
            (
                obstruction
                for obstruction in obstructions
                if _input_evidence_is_obscured(bounds, obstruction)
            ),
            None,
        )
        if matched is None:
            continue
        states = dict(element.get("states") or {})
        states["fully_visible"] = False
        states["goal_relevant"] = False
        states.pop("focused", None)
        element["states"] = states
        evidence = [str(item) for item in (element.get("evidence") or [])]
        evidence.append("本地检测到顶部不透明遮挡，输入框完整边界不可审计")
        element["evidence"] = evidence[:6]
        changed = True
    if not changed:
        return scene
    overlays = [str(item) for item in (value.get("overlays") or [])]
    note = "本地检测到顶部不透明视觉遮挡；交叠输入证据已失败关闭"
    if note not in overlays:
        overlays.append(note)
    value["overlays"] = overlays
    return UIScene.from_dict(
        value,
        coordinate_scale=1.0,
        stable_override=True,
        fingerprint_override=fingerprint,
    )


def _json_only_system_message() -> dict[str, str]:
    return {
        "role": "system",
        "content": (
            "你是只读页面观察器。只输出一个语法完整的JSON对象；禁止Markdown、解释、"
            "思考过程、代码围栏、JSON字符串套壳或对象前后的任何文字。"
        ),
    }


PREFILLED_INPUT_OBSERVATION_RULE = (
    "输入框可能为空，也可能已经含有文字；预填充且未聚焦时可以没有光标或占位提示。"
    "当一个有清晰独立边界的横向矩形内含查询/地址/表单文字，并带有边界独立的尾部功能控件"
    "（例如搜索、提交、清除、语音或扫描图标）时，"
    "这组结构本身就是role=input的可靠视觉证据，不得仅因没有光标而降级成text或container；"
    "尾部功能控件必须作为另一个控件观察，不能把输入框和功能控件合成横幅。这个判断只报告"
    "页面事实，绝不表示可以激活尾部控件。框内文字的内容或主题不能改变控件角色；其他没有"
    "上述成组结构的带文字区域仍不得仅因含有文字就被认作输入框。"
)

INPUT_VALUE_AND_MODE_OBSERVATION_RULE = (
    "role=input且框内文字清晰可读时，必须在states.value中逐字填写当前可见文字；空框写空字符串，"
    "看不清才省略value，禁止根据目标补写。软键盘可见时还必须在states.keyboard_layout写"
    "qwerty、numeric、symbol或unknown，并在states.keyboard_input_mode写direct_latin、"
    "chinese_pinyin或unknown。QWERTY只描述按键排列，绝不等于英文直输：画面出现中文候选、"
    "拼音分词撇号或明确中文模式时必须写chinese_pinyin；只有明确显示英文/Latin直输模式时才能写"
    "direct_latin；看不清写unknown。这些都只是画面事实，不授权输入。"
)

KEYBOARD_MODE_SWITCH_OBSERVATION_RULE = (
    "若键盘底部清楚可见独立的"
    "中/英模式切换键，必须另建role=button元素，meaning写switch_keyboard_input_mode，label逐字抄"
    "可见键面文字，states写keyboard_input_mode_switch:true、current_mode和target_mode；不确定当前"
    "模式或切换方向时不得编造该元素。字母、数字、退格、回车等普通键不得进入elements；"
    "它们不是通用语义动作目标，键盘布局、输入模式和按键几何由后续独立全帧输入结构审计负责。"
)

LOCAL_TEXT_CLEAR_OBSERVATION_RULE = (
    "若非空输入框内部或紧邻右侧清楚可见独立的圆形×/清空图标，必须另建role=button或icon元素，"
    "meaning写clear_local_text，states写local_text_clear:true，label必须逐字写图标本身的×/✕/✖/x；"
    "若看不清真实叉号图形或只能自由描述为叉号，就不得标记local_text_clear。只框该图标自身，不能与输入框合并，"
    "也绝不能把键盘退格键/删除键标成local_text_clear。页面右侧的文字‘取消’/cancel是取消编辑或"
    "退出控件，不是本地清空图标；必须meaning=cancel且goal_relevant:false，绝不能标成clear_local_text。"
)

INPUT_VALUE_OBSERVATION_RULE = (
    INPUT_VALUE_AND_MODE_OBSERVATION_RULE
    + KEYBOARD_MODE_SWITCH_OBSERVATION_RULE
    + LOCAL_TEXT_CLEAR_OBSERVATION_RULE
)

SYSTEM_UI_OBSERVATION_RULE = (
    "system_ui必须始终存在，且只允许immersive_or_fullscreen和"
    "navigation_bar_visible两个字段。每个值只能是true、false或字符串unknown："
    "只有画面明确证明时才写布尔值，裁切、遮挡、模糊或无法排除时必须写unknown。"
    "这是系统UI的只读事实，不是完成判断。navigation_bar、system_navigation_bar或"
    "system_nav_bar绝不得写入elements，即使它部分可见或与目标相关。"
)

CAMERA_ALIGNMENT_OBSERVATION_RULE = (
    "camera_alignment in the compact scene is descriptive and can never authorize "
    "hardware. camera_layout_orientation describes the supplied canvas only "
    "and must be portrait, landscape, or square. phone_content_rotation describes "
    "how the phone App/system axes appear: upright, rotated_90, "
    "rotated_180, rotated_270, or unknown. Inspect only the physical phone display; "
    "seller-controller PX/MM readouts, colored borders, and bottom action/orientation "
    "buttons are external chrome and never phone evidence. Black or sparse App content "
    "does not lower alignment confidence "
    "when visible phone text or system structure establishes its axes. Evidence must "
    "contain one or two short non-control strings and no coordinates, actions, or bounds."
)


def _orientation_audit_prompt() -> str:
    return f"""
This is an independent read-only camera/phone-axis audit. The three images are
the same stable frame: original, ROTATE_90, ROTATE_270; they are not temporal.
Classify ONLY Image 1, the original stable frame, relative to Image 1's own
canvas. Images 2 and 3 are derived orientation references only. They may help
identify the phone-content axes, but they are never classification targets.
Never report the rotation of Image 2 or Image 3, never report which reference
looks upright, and never report the transform that would make Image 1 upright.
For example, if Image 1 is already upright, return "upright" even though Images
2 and 3 show rotated copies. phone_content_rotation must always describe how
the phone App/system axes appear inside Image 1 as supplied.
Judge only the physical phone display. Seller-controller PX/MM text, borders,
orientation buttons and bottom controls are external chrome and forbidden evidence.
Black or sparse App content does not reduce confidence when phone/system text or
structure establishes axes. Evidence must contain one or two phone-only facts,
each at most 60 characters. Describe only visible text orientation or phone UI
structure. Never mention what a person or controller could do, and never mention
coordinates, actions, bounds, the seller controller, robot arm, or external control.
Return exactly this JSON object and no Markdown:
{{"protocol_version":"{ORIENTATION_AUDIT_PROTOCOL_VERSION}",
"phone_content_rotation":"upright|rotated_90|rotated_180|rotated_270|unknown",
"confidence":0.0,"evidence":["one or two short phone-only facts"]}}
""".strip()


def _orientation_audit_retry_prompt(error: Exception) -> str:
    return f"""
The preceding independent read-only orientation audit was rejected before any
physical action because its evidence wording was not a phone-only visual fact.
Error summary: {str(error)[:180]}
Re-observe the same three derived orientation views independently. Classify only
IMAGE 1. Return exactly this JSON object and no Markdown or extra fields:
{{"protocol_version":"{ORIENTATION_AUDIT_PROTOCOL_VERSION}",
"phone_content_rotation":"upright|rotated_90|rotated_180|rotated_270|unknown",
"confidence":0.0,"evidence":["short phone-only visual fact"]}}
Evidence must contain one or two strings, each at most 60 characters. State only
the orientation of visible phone text or phone UI structure. Do not mention any
tap, click, press, swipe, drag, execution, suggestion, coordinate, bound, PX/MM,
seller controller, robot arm, external control, or possible action. JSON only.
""".strip()


def _orientation_evidence_format_error(error: Exception) -> bool:
    return str(error) in {
        "方向凭据只读证据无效。",
        "方向凭据包含坐标、动作或外部控制端证据。",
    }


def _credential_from_orientation_audit(
    *,
    raw: str,
    device_id: str,
    scene_fingerprint: str,
    frame: Image.Image,
) -> OrientationCredential:
    payload = _parse_orientation_audit(raw)
    return _mint_audited_credential(
        device_id=device_id,
        scene_fingerprint=scene_fingerprint,
        frame=frame,
        phone_content_rotation=payload["phone_content_rotation"],
        confidence=payload["confidence"],
        evidence=tuple(payload["evidence"]),
    )


def _orientation_audit_diagnostic_payload(raw: str) -> dict[str, Any]:
    """Keep only bounded protocol facts; never retain raw evidence or authority."""

    def value_type(value: Any, *, missing: object) -> str:
        if value is missing:
            return "missing"
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, (int, float)):
            return "number"
        if isinstance(value, str):
            return "string"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        return "other"

    missing = object()
    result: dict[str, Any] = {
        "payload_object_found": False,
        "protocol_version_match": False,
        "rotation_valid": False,
        "confidence_valid": False,
        "confidence_type": "missing",
        "evidence_value_type": "missing",
        "unexpected_fields_count": 0,
        "has_unexpected_fields": False,
    }
    try:
        payload = _extract_json_object(raw)
    except Exception:
        return result
    if not isinstance(payload, dict):
        return result
    result["payload_object_found"] = True
    allowed = {
        "protocol_version", "phone_content_rotation", "confidence", "evidence"
    }
    unexpected_count = sum(1 for key in payload if key not in allowed)
    result["unexpected_fields_count"] = unexpected_count
    result["has_unexpected_fields"] = unexpected_count > 0
    result["protocol_version_match"] = (
        payload.get("protocol_version", missing)
        == ORIENTATION_AUDIT_PROTOCOL_VERSION
    )
    rotation = payload.get("phone_content_rotation", missing)
    valid_rotations = {
        "upright", "rotated_90", "rotated_180", "rotated_270", "unknown"
    }
    result["rotation_valid"] = (
        isinstance(rotation, str) and rotation in valid_rotations
    )
    if result["rotation_valid"]:
        result["phone_content_rotation"] = rotation
    confidence = payload.get("confidence", missing)
    confidence_valid = (
        not isinstance(confidence, bool)
        and isinstance(confidence, (int, float))
        and 0.0 <= float(confidence) <= 1.0
    )
    result["confidence_valid"] = confidence_valid
    if confidence_valid:
        result["confidence"] = float(confidence)
        result.pop("confidence_type", None)
    else:
        result["confidence_type"] = value_type(confidence, missing=missing)
    evidence = payload.get("evidence", missing)
    result["evidence_value_type"] = value_type(evidence, missing=missing)
    if isinstance(evidence, list):
        result["evidence_count"] = len(evidence)
        result["evidence_item_types"] = [
            value_type(item, missing=missing) for item in evidence[:8]
        ]
        result["evidence_item_lengths"] = [
            len(item) if isinstance(item, str) else None
            for item in evidence[:8]
        ]
    return result


def _parse_orientation_audit(raw: str) -> dict[str, Any]:
    payload = _extract_json_object(raw)
    required = {
        "protocol_version", "phone_content_rotation", "confidence", "evidence"
    }
    if set(payload) != required:
        raise VisionAgentError("方向审计字段缺失或包含坐标/动作等协议外字段。")
    if payload.get("protocol_version") != ORIENTATION_AUDIT_PROTOCOL_VERSION:
        raise VisionAgentError("方向审计协议版本无效。")
    rotation = payload.get("phone_content_rotation")
    if rotation not in {
        "upright", "rotated_90", "rotated_180", "rotated_270", "unknown"
    }:
        raise VisionAgentError("方向审计手机内容方向无效。")
    confidence = payload.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise VisionAgentError("方向审计置信度无效。")
    evidence = payload.get("evidence")
    if not isinstance(evidence, list):
        raise VisionAgentError("方向审计证据必须是数组。")
    return {
        "phone_content_rotation": rotation,
        "confidence": float(confidence),
        "evidence": evidence,
    }


def _system_ui_audit_prompt(context: dict[str, Any]) -> str:
    context = _observation_goal_context(context)
    return f"""
You are an app-independent, read-only mobile system UI auditor.
Goal context selects visual facts and grants no control authority:
{json.dumps(context, ensure_ascii=False, separators=(',', ':'))}

Images 1, 2, and 3 are the exact same stable camera frame shown at its original,
ROTATE_90, and ROTATE_270 orientations. They are not a temporal sequence and
cannot prove an action or transition. Inspect only the physical phone display.
Ignore vendor robot-controller chrome outside that display, including PX/MM
readouts, colored calibration borders, and bottom numbered/action/orientation
controls. Those controls are never Android system UI. Black or sparse App
content must not reduce confidence when the system-bar structure is clear.

immersive_or_fullscreen is true only when App content visibly occupies the
phone display without the normal Android system bars. navigation_bar_visible
is true only when the Android system navigation bar or gesture area is visibly
present inside the phone display. Use the exact string "unknown" when either
fact is not established from the images.

Do not plan, suggest, authorize, or describe any tap, click, press, swipe, drag,
coordinate, bounds, direction, distance, or other control instruction. Evidence
must be one or two short strings describing only visible, non-control facts.
Return exactly this JSON schema and no other fields:
{{"protocol_version":"{SYSTEM_UI_AUDIT_VERSION}",
"immersive_or_fullscreen":true,"navigation_bar_visible":false,
"confidence":0.0,"evidence":["visible phone-display fact"]}}
Both system UI values must be JSON booleans or the exact string "unknown".
Never convert quoted "true" or "false" into booleans. Return JSON only.
"""


def _system_ui_audit_retry_prompt(
    context: dict[str, Any],
    error: Exception,
) -> str:
    context = _observation_goal_context(context)
    return f"""
The preceding read-only system UI audit was rejected before any action.
Error summary: {str(error)[:260]}
Re-observe the same three orientation views independently.
Goal context is evidence selection only: {json.dumps(context, ensure_ascii=False, separators=(',', ':'))}
Return exactly this JSON object and no other fields:
{{"protocol_version":"{SYSTEM_UI_AUDIT_VERSION}",
"immersive_or_fullscreen":true,"navigation_bar_visible":false,
"confidence":0.0,"evidence":["short visible fact"]}}
Both facts must be JSON booleans when visually established or the exact string
"unknown" otherwise. evidence must contain one or two short JSON strings only:
no objects, coordinates, bounds, actions, suggestions, or controller controls.
Inspect only Android system UI inside the physical phone display. Ignore PX/MM,
colored calibration borders, bottom numbered/action/orientation controls, and
other vendor robot-controller chrome outside the phone display. JSON only.
"""


_FOREGROUND_APP_IDENTITY_PLACEHOLDERS = frozenset(
    {
        "current_foreground",
        "current_app",
        "foreground_app",
        "target_app",
        "active_app",
    }
)
_FOREGROUND_APP_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_RUNTIME_PACKAGE_APP_ID_PATTERN = re.compile(
    r"^(?:[a-z][a-z0-9_]*\.)+[a-z][a-z0-9_]*$"
)
_MIN_FOREGROUND_APP_IDENTITY_CONFIDENCE = 0.90


def _is_foreground_app_identity_placeholder(value: str) -> bool:
    return str(value or "").strip().casefold() in (
        _FOREGROUND_APP_IDENTITY_PLACEHOLDERS
    )


def _is_runtime_package_app_identity(value: str) -> bool:
    """Return whether an observed identity is an opaque runtime package name.

    The compact observer may report a concrete Android package while DeepSeek
    uses a short semantic App id.  A package is real scene evidence, but it is
    not directly comparable with that semantic id.  Keep this shape-only so no
    App name or package mapping can become action authority.
    """

    return bool(
        _RUNTIME_PACKAGE_APP_ID_PATTERN.fullmatch(
            str(value or "").strip().casefold()
        )
    )


def _needs_foreground_app_identity_audit(
    scene: UIScene,
    context: dict[str, Any],
) -> bool:
    """Audit unresolved identity only when structured page facts support it."""

    if _current_surface_input_does_not_require_app_identity(context):
        return False
    foreground = str(scene.foreground_app_id or "").strip().casefold()
    if _is_foreground_app_identity_placeholder(foreground):
        return True
    target_app = str(context.get("app_id") or "").strip().casefold()
    named_target = bool(
        target_app
        and target_app != "unknown"
        and not _is_foreground_app_identity_placeholder(target_app)
    )
    if foreground == "unknown":
        return named_target
    if not named_target or foreground == target_app:
        return False
    if _is_runtime_package_app_identity(foreground):
        return True
    return _scene_structurally_names_target_app(
        scene,
        target_app_id=target_app,
        target_app_name=str(context.get("app_name") or "").strip(),
    )


def _current_surface_input_does_not_require_app_identity(
    context: dict[str, Any],
) -> bool:
    """Keep App-brand discovery out of an exact typed current-surface task."""

    if not _goal_requests_input(context):
        return False
    if str(context.get("app_id") or "").strip().casefold() != "current_surface":
        return False
    entities = context.get("entities")
    if not isinstance(entities, dict):
        return False
    target_apps = entities.get("target_apps", [])
    if not isinstance(target_apps, list) or any(
        isinstance(item, str) and item.strip() for item in target_apps
    ):
        return False
    focused = _active_subgoal_visual_context(context)
    goal_entities = (
        focused.get("goal_entities")
        if focused is not context and isinstance(focused, dict)
        else entities
    )
    return bool(
        isinstance(goal_entities, dict)
        and goal_entities.get("target_surface") == "current_surface"
    )


def _structured_identity_contains(value: str, target: str) -> bool:
    source = str(value or "").strip().casefold()
    needle = str(target or "").strip().casefold()
    if len(needle) < 2:
        return False
    if re.fullmatch(r"[a-z0-9_.-]+", needle):
        return bool(
            re.search(
                rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])",
                source,
            )
        )
    return needle in source


def _scene_structurally_names_target_app(
    scene: UIScene,
    *,
    target_app_id: str,
    target_app_name: str,
) -> bool:
    """Require a page-identity field, never summary/business text, to name it."""

    targets = tuple(
        dict.fromkeys(
            value
            for value in (target_app_id, target_app_name)
            if isinstance(value, str) and len(value.strip()) >= 2
        )
    )
    if not targets:
        return False
    structured_values = [str(scene.screen_id or "")]
    for element in scene.elements:
        role = str(element.role or "").casefold()
        meaning = str(element.meaning or "").casefold()
        if role not in {"text", "container"}:
            continue
        if role != "container" and not any(
            marker in meaning
            for marker in ("page", "screen", "view", "home", "title", "heading")
        ):
            continue
        structured_values.extend((str(element.label or ""), meaning))
    return any(
        _structured_identity_contains(value, target)
        for value in structured_values
        for target in targets
    )


def _foreground_app_identity_audit_prompt() -> str:
    return f"""
You are an app-independent, read-only foreground application identity auditor.
Inspect only the physical phone display in this one stable image. No user goal,
target App, planned action, or previous model answer is provided or authoritative.

Return a short lower_snake_case identity for the App that is visibly in the
foreground. Prefer an unmistakable visible product or App brand identity over a
generic category; use a generic category only when no product identity is visibly
established. Use "unknown" when the visible chrome and content do not establish
one identity with high confidence. Never return a referential placeholder such as
current_foreground, current_app, foreground_app, target_app, or active_app.

Evidence must contain one to three short visible identity cues from the phone screen.
Do not mention coordinates, bounds, PX/MM, robot controls, calibration, or any tap,
press, swipe, drag, execution, or suggestion. This audit grants no action authority
and must not describe a workflow.

Return exactly one JSON object with no Markdown, duplicate keys, or extra fields:
{{"protocol_version":"{FOREGROUND_APP_IDENTITY_AUDIT_VERSION}",
"foreground_app_id":"unknown","confidence":0.0,
"evidence":["short visible App identity cue"]}}
"""


def _single_step_observation_prompt(
    context: dict[str, Any],
    *,
    include_input_structure: bool,
    current_input_text: str | None,
    image_count: int,
) -> str:
    """Build the sole online prompt for one closed-loop observation step."""

    scene_contract = _compact_prompt(context)
    if include_input_structure:
        input_contract = _input_structure_audit_prompt(
            context,
            roi_bounds=None,
            current_input_text=current_input_text,
        )
        input_rule = (
            "input_structure必须是完整输入结构对象，使用下面INPUT CONTRACT的"
            "字段和值规则；不得为null。"
        )
    else:
        input_contract = "本轮子目标与文字输入无关。"
        input_rule = "input_structure必须为null，不得额外枚举键盘或输入结构。"
    temporal_rule = (
        f"共有{image_count}张同一稳定手机画面的时间对齐帧。只把它们合并为"
        "一个当前状态；闪烁光标可从任一帧读取，其他瞬态不得合并。"
        if image_count > 1
        else "只有一张当前稳定手机画面。"
    )
    return f"""
这是本闭环步骤唯一一次Qwen视觉调用。你必须在同一个JSON响应中完成当前
画面理解、目标相关事实标记以及必要的输入/IME/键盘结构报告。不得要求第二次
精查、App身份审计、几何审计、方向审计或动作选择调用；不确定时保留unknown、
省略候选或降低confidence。你只报告事实，不输出动作、计划或坐标点击建议。
{temporal_rule}

下面的SCENE CONTRACT和INPUT CONTRACT沿用既有字段语义。它们各自末尾的
“只返回/Return exactly”示例仅说明对应内层对象，不是本轮顶层输出格式。

--- SCENE CONTRACT ---
{scene_contract}

--- INPUT CONTRACT ---
{input_contract}

最终且唯一有效的顶层格式如下，禁止Markdown、重复键和任何额外字段：
{{"protocol_version":"{SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION}",
"scene":{{"protocol_version":"{UI_SCENE_PROTOCOL_VERSION}",
"foreground_app_id":"unknown","screen_id":"unknown","summary":"",
"system_ui":{{"immersive_or_fullscreen":"unknown","navigation_bar_visible":"unknown"}},
"camera_alignment":{{"camera_layout_orientation":"portrait",
"phone_content_rotation":"unknown","confidence":0.0,"evidence":[]}},
"elements":[],"overlays":[],"stable":true,"confidence":0.0,"fingerprint":""}},
"input_structure":null}}
{input_rule}
scene.states.goal_relevant是本次单步响应对目标相关可见事实的唯一标记；本地只会
从由它和canonical目录共同证明的唯一候选中确定下一动作，绝不再请求Qwen选择。
"""


def _parse_single_step_observation_envelope(
    raw: str,
    *,
    input_structure_required: bool,
) -> dict[str, Any]:
    """Parse one fused response without any remote repair or resampling."""

    try:
        payload = _extract_json_object(raw)
        # Flat scene JSON remains a valid one-call response for non-input
        # observations.  This is a shape normalization only; it never enables
        # a second request or restores any former audit authority.
        if payload.get("protocol_version") == UI_SCENE_PROTOCOL_VERSION:
            if input_structure_required:
                raise UISceneError("输入子目标的单次响应缺少input_structure。")
            return {"scene": payload, "input_structure": None}
        required = {"protocol_version", "scene", "input_structure"}
        if set(payload) != required:
            missing = sorted(required - set(payload))
            extra = sorted(set(payload) - required)
            details = []
            if missing:
                details.append("缺少字段：" + ", ".join(missing))
            if extra:
                details.append("包含协议外字段：" + ", ".join(extra))
            raise UISceneError("单步观察封装结构无效；" + "；".join(details))
        if payload["protocol_version"] != SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION:
            raise UISceneError("单步观察协议版本不匹配。")
        if not isinstance(payload["scene"], dict):
            raise UISceneError("单步观察scene必须是对象。")
        input_payload = payload["input_structure"]
        if input_structure_required and not isinstance(input_payload, dict):
            raise UISceneError("输入子目标必须在同一响应返回input_structure对象。")
        if not input_structure_required and input_payload is not None:
            raise UISceneError("非输入子目标的input_structure必须为null。")
        return payload
    except (UISceneError, ValueError, TypeError) as exc:
        raise VisionAgentError(f"单步完整观察结果不符合协议：{exc}") from exc


def _compact_prompt(context: dict[str, Any]) -> str:
    privacy_note = (
        "本轮是坐标无关的Android系统Home观察。中央App内容已由本地固定遮罩隐藏；"
        "只能根据保留的手机画布边缘和底部Android系统导航结构报告unknown场景、画布方向和稳定性，"
        "不得猜测App、正文或元素。"
        if _goal_requests_coordinate_free_system_home(context)
        else ""
    )
    context = _observation_goal_context(context)
    if _goal_requests_keyboard_mode_switch(context):
        keyboard_switch_rule = (
            " 当前子目标明确要求切换键盘输入模式；本轮快速观察不得在elements中报告或定位"
            "任何模式切换键。后续独立全帧输入结构审计是模式、方向和模式键几何的唯一权威。"
            "普通输入框和键盘可见事实仍可报告，但不得据此建议动作。"
        )
    else:
        keyboard_switch_rule = KEYBOARD_MODE_SWITCH_OBSERVATION_RULE
    input_observation_rule = (
        INPUT_VALUE_AND_MODE_OBSERVATION_RULE
        + keyboard_switch_rule
        + LOCAL_TEXT_CLEAR_OBSERVATION_RULE
    )
    return f"""
你是通用手机页面观察器，只报告画面事实，不规划也不执行动作。
用户目标只用于选择需要读清的控件，不能让你幻读：
{json.dumps(context, ensure_ascii=False, separators=(',', ':'))}
{privacy_note}

用最短JSON报告：当前前台App、页面类型、最上层弹层，以及与目标直接相关的可见控件。
规则：
1. 桌面写 launcher；不确定写 unknown。不得把目标App当成当前App，也不得把
   current_foreground、current_app、foreground_app、target_app 或 active_app 等引用占位符
   写成foreground_app_id；该字段只能来自当前画面的视觉身份。
2. elements最多{MAX_COMPACT_ELEMENTS}个。必须先报告目标相关控件和当前输入框，
   再报告关闭/返回与必要导航；省略新闻、商品、图片、标签组等无关内容。
3. bounds使用0..1000的[left,top,right,bottom]，必须只框真实清晰控件。0和1000分别代表
   原图四边；禁止复制原图像素坐标（例如810x1515画面的y=1130），任何边界超出0..1000就省略该元素。
4. role仅限button/icon/input/text/tab/toggle/image/list_item/dialog/keyboard_key/container/unknown。
   container只表示承载其他内容的分组、布局区或目标区域；四边独立、可单独识别的色块、卡片、图片
   或控件不得写container，应按可见形态写image/list_item/button。可见文字或外观明确证明的移动源
   与目标区域必须分别建元素，不能合成一个container。目标相关元素必须在states中逐项报告
   fully_visible:true/false；只有整个轮廓均在原图内且无遮挡时才可写true。
5. meaning用lower_snake_case。与目标直接相关的控件在states中写goal_relevant:true。
6. 每个element的evidence最多一条不超过40个字的画面短文字或明确外观。
   summary不超过60个字。看不清就降低confidence或省略元素。
7. 禁止action、plan、step、tap、swipe、command、coordinates等动作字段。
   overlays只允许简短字符串名称；任何带边界、角色或ID的可交互候选必须放入elements，
   不得把对象放入overlays。
8. 场景confidence只评价当前画面本身是否清楚、稳定、可描述，不评价目标是否已完成或目标控件
   是否存在。清晰稳定的页面即使没有目标控件，也应保持与画面质量一致的高confidence并返回空
   elements；只有模糊、遮挡、过渡或无法判断页面事实时才降低confidence。
9. {PREFILLED_INPUT_OBSERVATION_RULE}
10. {input_observation_rule}
11. {SYSTEM_UI_OBSERVATION_RULE}
12. {CAMERA_ALIGNMENT_OBSERVATION_RULE}
13. 如果目标尚未出现，而当前画面明确是列表或信息流，并且原图边缘能看见只露出一部分的后续
    列表项/卡片，必须在summary中简短记录“对应边缘存在部分可见的后续内容，列表仍在延伸”。
    这是只读滚动线索，不得猜测被裁切项就是目标，不得给动作建议；被裁切元素不得标成可操作目标。
14. 对分步流程、时间线或其他结构化长页面，如果原图中有连续引导轨、连接线或内容轨道明确延伸并
    接触视口边缘，必须在summary记录“对应边缘存在明确的页面延续标记，内容仍可继续浏览”。只有线条
    确实属于页面内容且连续到边缘时才能报告；装饰线、手机边框和机械臂控制器标线不算。该事实同样
    只是只读滚动线索，不能猜测边缘之外的目标或给出动作建议。
15. 如果目标用“从上往下第N项/列表第N项/first、second、Nth item”等序数指定同一列表内的
    可见条目，必须把目标条目及其之前所有同列、同类、完整可见的兄弟条目分别写入elements，
    每项逐字抄录label并紧框自身；只有目标条目写goal_relevant:true，前序证明项写false。
    序数必须按这些条目的垂直中心从上到下比较，不能根据文字含义猜测。若N超过elements上限、
    任一前序项不可见/被遮挡/无法同列绑定，或不能逐项证明顺序，就不得把任何候选标成目标相关，
    并在summary说明序数证据不足。
16. 如果目标要求看清、读取或核对当前/下一页的标题、题头、heading或title，必须优先报告唯一清晰
    的页面主标题：role=text、meaning=page_title、label逐字抄录、goal_relevant:true，并明确
    fully_visible。清晰主标题可直接作为screen_id；普通正文、卡片说明、按钮文字和浏览器标题栏
    不能冒充页面主标题。看不清、存在多个同级主标题或标题不完整时保持screen_id=unknown。
17. 如果当前目标明确要求滑动/上划/下划/左划/右划，并且原图能明确证明一个可滚动视口，必须把
    该视口作为一个role=container元素报告。只有以下任一视觉条件成立才算证明：同一视口内至少两个
    重复同类条目按同一轴排列；或相关边缘存在被裁切的后续内容；或属于页面内容的连续轨道明确接触
    相关边缘。该container必须紧框完整可见的内容视口，states必须逐项包含
    goal_relevant:true、fully_visible:true、scrollable:true、scroll_axis:"vertical"或"horizontal"，
    evidence必须说明实际看见的重复结构或边缘延续。单张卡片、工具栏、页面边框、目标动作文字本身
    都不能证明scrollable；无法证明时不得输出该状态，也不得猜测。

只返回下列完整JSON，不要Markdown：
{{"protocol_version":"{UI_SCENE_PROTOCOL_VERSION}","foreground_app_id":"unknown",
"screen_id":"unknown","summary":"当前画面短描述","system_ui":{{"immersive_or_fullscreen":"unknown",
"navigation_bar_visible":"unknown"}},"camera_alignment":{{"camera_layout_orientation":"portrait",
"phone_content_rotation":"unknown","confidence":0.0,"evidence":[]}},"elements":[],"overlays":[],
"stable":true,"confidence":0.0,"fingerprint":""}}
每个element只允许：
{{"element_id":"e1","role":"button","meaning":"open_search","label":"搜索",
"bounds":[0,0,1000,1000],"confidence":0.0,"states":{{"goal_relevant":true}},"evidence":[]}}
"""


def _targeted_prompt(
    context: dict[str, Any],
    *,
    first_scene: dict[str, Any],
    roi_bounds: tuple[int, int, int, int] | None = None,
) -> str:
    context = _observation_goal_context(context)
    # Keep the first scene short to avoid anchoring the model with many labels.
    compact_scene = {
        "foreground_app_id": first_scene.get("foreground_app_id"),
        "screen_id": first_scene.get("screen_id"),
        "summary": first_scene.get("summary"),
        "system_ui": first_scene.get("system_ui"),
        "overlays": first_scene.get("overlays"),
        "confidence": first_scene.get("confidence"),
    }
    return f"""
你是通用手机页面观察器。快速观察没有找到足够明确的目标相关控件，现在只做目标精查，仍然不能规划或执行动作。
用户目标：{json.dumps(context, ensure_ascii=False, separators=(',', ':'))}
快速观察摘要：{json.dumps(compact_scene, ensure_ascii=False, separators=(',', ':'))}
{_roi_observation_note(roi_bounds)}

重新检查原图中与目标直接相关的文字、图标、输入框、列表项和最上层弹层。
只保留最多{MAX_COMPACT_ELEMENTS}个最相关元素；目标元素必须states.goal_relevant=true。看不清或不唯一就不要输出，
不得为了填满elements枚举与目标无关的导航标签、工具栏项或正文；
并降低confidence。bounds的x和y必须分别按原图宽、高独立归一化到0..1000：
左/上边为0，右/下边为1000。竖图不是以宽度1000等比缩放后的长方形坐标系；
任何y>1000都说明坐标系错了，必须省略该元素，不得截断或换算。只框元素自身。
禁止任何动作、计划或建议字段。
目标相关元素既包括已经满足完成条件的可见结果，也包括画面上清楚可见、能使该结果进入视野
的入口控件；这里只报告控件事实，不建议也不授权使用它。
置信度只评价当前画面观察本身是否可靠，不能因为目标尚未完成而降低；例如清晰桌面上唯一目标
应用入口可形成高可信观察，即使应用尚未打开。模糊、遮挡或不唯一时仍必须降低，禁止虚增。
目标相关控件确实不存在时返回空elements，但只要页面事实清楚稳定，confidence仍应保持高值；
不得因为系统级动作没有屏内按钮、或因为未找到目标控件，就把清晰页面写成低置信。
如果目标尚未出现，而当前画面明确是列表/信息流且原图边缘能看到部分可见的后续列表项或卡片，
summary_addendum必须记录“对应边缘存在部分可见的后续内容，列表仍在延伸”。这只是只读页面事实，不能猜测
被裁切项就是目标，不能给动作建议，也不能把被裁切项写成可操作目标。
如果当前是分步流程、时间线或结构化长页面，且属于页面内容的连续引导轨、连接线或内容轨道明确延伸
并接触原图边缘，summary_addendum必须记录“对应边缘存在明确的页面延续标记，内容仍可继续浏览”。装饰线、
手机边框和机械臂控制器标线不算；不得猜测边缘外是什么，也不得把该标记写成可操作目标。
如果当前目标明确要求滑动/上划/下划/左划/右划，并且原图能明确证明同一视口内至少两个重复同类
条目按同一轴排列，或相关边缘存在被裁切后续内容/连续内容轨，必须把完整可见内容视口报告为
role=container，states逐项包含goal_relevant:true、fully_visible:true、scrollable:true以及
scroll_axis:"vertical"或"horizontal"，evidence说明实际视觉证据。单张卡片、工具栏、页面边框或
目标文字本身不能证明scrollable；证据不足就省略，绝不能为了产生滑动候选而猜测。
若目标以序数指定列表条目，必须把目标及其之前所有同列、同类、完整可见兄弟项分别写入elements，
逐字抄录label并紧框自身；只把按垂直中心从上到下排序后位于指定序位的条目标成goal_relevant:true，
前序证明项写false。缺少任一前序项、超过{MAX_COMPACT_ELEMENTS}个元素或无法证明同列顺序时不得猜测目标。
若目标要求读取当前或下一页标题，必须优先返回唯一页面主标题元素，使用role=text、
meaning=page_title、逐字label、goal_relevant:true并明确fully_visible；普通正文、按钮文字、
卡片说明和浏览器标题栏都不是页面主标题。
若目标是图标且高清局部内存在两个或以上相邻图标，必须逐个区分图标语义：一个element只能紧框一个
完整图标，绝不能把工具栏、图标组或相邻图标合成同一bounds。目标图标与相邻非目标图标可明确区分时，
只把目标写goal_relevant:true，相邻图标写false或省略；证据必须说明看见的目标字面图形以及与相邻图标
的区别。fully_visible只评价目标图标自身在完整原图中的四边是否都可见，不能因为工具栏贴近画面边缘就
把完整图标写false，也不能因为高清局部放大而把原图中真实被裁切的图标写true。无法逐个紧框、无法
区分语义或目标自身任一边被裁切时，不得输出可操作目标。
    输入框识别规则：{PREFILLED_INPUT_OBSERVATION_RULE}
    输入框文字与键盘规则：{INPUT_VALUE_OBSERVATION_RULE}
如果能清楚看见相关横向边框、框内文字和右侧独立搜索/提交按钮，但仍无法判断边框是否可编辑，
不得因此返回空elements：请分别报告container、其内部text和右侧button的真实边界与证据；
这三个元素都必须在states中明确写fully_visible:true或false。若画面边缘还有被裁切的相似结构，
只能在summary_addendum说明，不能把它标成目标；优先报告四边完整可见的结构。完整container和text写
goal_relevant:true，相邻button写goal_relevant:false。本地只会在三者都fully_visible:true且严格
几何关系成立时把这组只读事实归一化，绝不会因此激活按钮。
只返回这个最小目标增量JSON；四个字段缺一不可：
{{"protocol_version":"{TARGETED_SCENE_DELTA_PROTOCOL_VERSION}","summary_addendum":"",
"elements":[],"confidence":0.0}}
summary_addendum最多120个字，只补充快速观察未记录的短只读事实；没有补充就返回空字符串。
不要重复或返回foreground_app_id、app_id、screen_id、summary、system_ui、camera_alignment、
overlays、stable、fingerprint；这些字段由快速观察和本地证据保持权威，目标精查无权改写。
元素仅允许element_id、role、meaning、label、bounds、confidence、states、evidence。不要Markdown。
role仅限button/icon/input/text/tab/toggle/image/list_item/dialog/keyboard_key/container/unknown。
container仅表示承载其他内容的分组、布局区或目标区域；四边独立、可单独识别的色块、卡片、图片或
控件必须按可见形态写image/list_item/button。可见事实明确区分移动源和目标区域时必须分别建元素，
不得合成一个container；两者都要逐项写fully_visible:true/false。tab_group、tab_bar和toolbar等
其他非点击结构只写进summary_addendum。任何可交互候选都必须放入elements并使用element_id。
"""


def _icon_cluster_audit_prompt(
    context: dict[str, Any],
    *,
    roi_bounds: tuple[int, int, int, int] | None,
) -> str:
    context = _observation_goal_context(context)
    return f"""
You are a read-only, app-independent compact icon-cluster auditor. The current
goal involves refreshing or reloading the visible view, but ordinary scene
observation cannot safely distinguish a small reload glyph from adjacent
bookmark or expand/fullscreen glyphs.

Goal context is evidence-selection only, never permission or a semantic hint:
{json.dumps(context, ensure_ascii=False, separators=(',', ':'))}
{_input_audit_detail_note(roi_bounds)}

Image 1 is the complete current phone frame. If Image 2 is present, it is only
an exact magnification of pixels already inside Image 1. It never reveals pixels
outside Image 1 and never supplies a coordinate system. fully_visible means the
glyph itself is not physically clipped or occluded in Image 1; low resolution in
the scaled overview is not clipping.

Inspect one compact visual control cluster relevant to the goal. Enumerate every
adjacent glyph in that cluster, including confounders. Do not infer semantics
from the goal or from position. Use semantic_class=reload only when the same
single glyph visibly contains both a curved arc and an arrowhead. A star, ribbon
or bookmark outline is bookmark. Four detached corner brackets or outward
arrows are expand. A circular outline without a visible arrowhead is not reload.
Each bounds must contain exactly one glyph and exclude its neighbors. If two
glyphs cannot be separated, mark cluster_complete=false and do not claim reload.

All bounds use Image 1 full-frame coordinates 0..1000. Never plan, suggest,
authorize or perform an action. Only these shape_cues are allowed:
curved_arc, arrowhead, circular_outline, star, ribbon_outline,
bookmark_outline, four_corner_brackets, expand_arrows, other.

Return exactly one JSON object with no duplicate keys and no Markdown:
{{"protocol_version":"{ICON_CLUSTER_AUDIT_VERSION}",
"cluster_complete":true,"cluster_bounds":[0,0,1000,1000],
"controls":[{{"control_id":"control-1","semantic_class":"reload|bookmark|expand|other",
"bounds":[0,0,1000,1000],"confidence":0.0,"fully_visible":true,
"single_glyph":true,"shape_cues":["curved_arc","arrowhead"]}}]}}

Top-level fields and control fields must match the schema exactly. controls may
be empty only when no trustworthy cluster is visible; then cluster_complete must
be false and cluster_bounds must be null.
"""


def _icon_cluster_localization_prompt() -> str:
    return f"""
You are the second, independent read-only localization check for one compact
icon cluster. Image 1 is a pixel-exact crop around the cluster selected from the
complete phone frame by the previous broad audit. The local controller adds a
small context margin so edge glyphs are not falsely treated as clipped. Its coordinate system is local to this crop:
left/top=0 and right/bottom=1000. The local controller will map your bounds back
to the complete frame; never copy or guess full-frame coordinates.

Enumerate every adjacent glyph actually visible in this crop. Do not infer a
glyph from the goal or its position. semantic_class=reload requires one single
glyph that visibly contains both curved_arc and arrowhead. Bookmark/star/ribbon
and expand/four-corner glyphs are confounders. Each bounds must tightly contain
exactly one glyph and exclude neighbors. If the crop does not contain the whole
cluster, the glyph is too blurry to localize, or a control cannot be separated,
set cluster_complete=false and do not claim reload.

Never plan, suggest, authorize or perform an action. Only these shape_cues are
allowed: curved_arc, arrowhead, circular_outline, star, ribbon_outline,
bookmark_outline, four_corner_brackets, expand_arrows, other.

Return exactly one JSON object with no duplicate keys and no Markdown:
{{"protocol_version":"{ICON_CLUSTER_AUDIT_VERSION}",
"cluster_complete":true,"cluster_bounds":[0,0,1000,1000],
"controls":[{{"control_id":"control-1","semantic_class":"reload|bookmark|expand|other",
"bounds":[0,0,1000,1000],"confidence":0.0,"fully_visible":true,
"single_glyph":true,"shape_cues":["curved_arc","arrowhead"]}}]}}
Top-level and control fields must match this schema exactly. controls may be
empty only when no trustworthy complete cluster is visible; then
cluster_complete=false and cluster_bounds=null.
"""


def _input_audit_literal_key_targets(
    context: dict[str, Any],
    *,
    current_input_text: str | None = None,
) -> tuple[str, ...]:
    """Return the bounded non-letter key whitelist for the active input goal."""

    entities = context.get("goal_entities")
    if not isinstance(entities, dict):
        entities = context.get("entities")
    text = None
    if isinstance(entities, dict):
        text = entities.get("active_input_transaction_text")
        if not isinstance(text, str) or not text:
            text = entities.get("input_text")
    if not isinstance(text, str):
        return ()
    if current_input_text is not None:
        try:
            next_step = plan_next_verified_input(text, current_input_text)
        except (ValueError, VerifiedTextTransactionError):
            next_step = None
        else:
            if next_step is None:
                return ()
            if next_step.kind == "literal_key" and next_step.segment not in {
                "\r",
                "\n",
            }:
                return (next_step.segment,)
            return ()
    targets: list[str] = []
    for character in text:
        if character in {"\r", "\n", "\t"} or character.isalpha():
            continue
        if character not in targets:
            targets.append(character)
        if len(targets) == 8:
            break
    return tuple(targets)


def _input_structure_audit_prompt(
    context: dict[str, Any],
    *,
    roi_bounds: tuple[int, int, int, int] | None,
    crop_local: bool = False,
    current_input_text: str | None = None,
) -> str:
    context = _observation_goal_context(context)
    literal_key_targets = _input_audit_literal_key_targets(
        context,
        current_input_text=current_input_text,
    )
    literal_keys_example = []
    if literal_key_targets:
        example_value = literal_key_targets[0]
        literal_keys_example = [
            {
                "value": example_value,
                "label": "Space" if example_value == " " else example_value,
                "key_kind": "space" if example_value == " " else "character",
                "bounds": [0, 0, 1000, 1000],
                "confidence": 0.0,
                "fully_visible": True,
            }
        ]
    literal_keys_example_json = json.dumps(
        literal_keys_example,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    active_field_id, active_field_label, active_multiline = (
        _goal_active_input_field(context)
    )
    target_text = (
        _goal_active_input_transaction_text(context)
        or _goal_explicit_input_text(context)
    )
    enter_required = False
    if target_text and current_input_text is not None and active_multiline:
        try:
            next_input_step = plan_next_verified_input(
                target_text,
                current_input_text,
            )
        except (ValueError, VerifiedTextTransactionError):
            next_input_step = None
        enter_required = bool(
            next_input_step is not None
            and next_input_step.kind == "literal_key"
            and next_input_step.segment == "\n"
        )
    if crop_local:
        if roi_bounds is None:
            raise ValueError("crop-local 输入审计必须绑定 ROI。")
        image_contract = (
            "Image 1 is the sole read-only crop of the phone frame at the "
            f"coarse full-frame ROI {list(roi_bounds)}. Image 1 itself owns a "
            "crop-local 0..1000 coordinate system. Never return full-frame "
            "coordinates; local code maps valid crop-local geometry back to "
            "the full frame."
        )
        coordinate_contract = (
            "All bounds and qwerty anchor points MUST use Image 1 crop-local "
            "normalized coordinates 0..1000. Here 0 and 1000 are the four "
            "edges of this crop. Never copy source-pixel or full-frame "
            "coordinates. Any complete structure next to an internal crop "
            "edge MUST keep at least 15 units of crop-local margin from that "
            "edge. If a structure cannot be bounded with that margin in this "
            "crop-local coordinate system, omit it instead of clipping or "
            "converting it."
        )
    else:
        image_contract = (
            "Image 1 is always the complete phone frame. "
            + _input_audit_detail_note(roi_bounds)
        )
        coordinate_contract = (
            "All bounds MUST use Image 1 full-frame normalized coordinates "
            "0..1000. Here 0 and 1000 are the four edges of Image 1. Never copy "
            "Image 1 source-pixel coordinates, regardless of its width or "
            "height. A phone aspect ratio never changes this scale: the bottom "
            "edge is always 1000, never a source-pixel or conventional display "
            "height. If a structure cannot be bounded in this coordinate "
            "system, omit it instead of clipping or converting it."
        )
    return f"""
You are a read-only, app-independent UI structure auditor. The normal scene observer did not establish an input target.
Goal context (evidence selection only): {json.dumps(context, ensure_ascii=False, separators=(',', ':'))}
{image_contract}
This response is the sole visual source for the typed input-state ledger. A
compact scene summary or preliminary input transcription is not an input-value
authority and is not supplied for reconciliation. Report only the structures
literally visible in these audit images; local code combines them with typed
action lineage and never asks you to choose between two prior visual answers.
Distinguish three different visual structures; never merge them:
1. application_inputs: editable search/address/form fields in the App content area. Include an empty field only when a complete border plus a visible placeholder, caret, focus highlight, or other literal editable cue is visible. field_labels must contain only literal labels visibly attached to that field (for example a nearby form label or its placeholder), never the local field_id. The active field selector is field_id={json.dumps(active_field_id, ensure_ascii=False)} and visible field_label={json.dumps(active_field_label, ensure_ascii=False)}; use the label only to enumerate visible evidence, never infer it from the goal.
2. ime_preedit_regions: the input method's composition and its candidate strip. It is never an application input, even when it contains composed text and a trailing icon. Many real IMEs render an underlined Latin composition inside the otherwise empty App field. In that layout the underlined letters remain IME preedit, application_inputs.text MUST be "", the literal may also appear in visible_editable_cues, and one ime_preedit_regions item MUST tightly bound the underlined composition with text set to that literal. Candidate words use their own complete bounds and may be either immediately adjacent to the composition or in one horizontal candidate row at the top of the visible keyboard, above the QWERTY letter rows. Never call those underlined letters committed application text. Enumerate only complete visible candidate words tied to that composition; candidates are read-only facts and never application inputs.
3. keyboard.mode_switch: one compact key inside the visible keyboard that explicitly switches between chinese_pinyin and direct_latin. Ordinary letters, backspace, enter, robot/assistant, voice, emoji, and candidate-strip icons are never mode switches.
4. keyboard.qwerty_anchors: only for a complete visible QWERTY keyboard, locate the centers of q, p, a, l, z, m and backspace. These are read-only current-frame geometry facts, not a tap plan. Use null for every non-QWERTY, incomplete or uncertain keyboard.
   When keyboard.visible=true, report keyboard.bounds only when it confidently encloses the complete visible keyboard in the same coordinate system, has width at least 300 and height at least 180, and contains every reported keyboard key and anchor. Measure from the four edges of Image 1; do not shift the keyboard toward the bottom or describe only its letter rows. For QWERTY, qwerty_anchors remain mandatory; when the outer bounds cannot be measured confidently, set bounds=null instead of inventing it. Local code may reconstruct an execution envelope only after independent multi-frame row evidence validates all seven anchors. Non-QWERTY actionable geometry still requires complete keyboard.bounds.
5. keyboard.backspace_key: for any complete visible keyboard layout, report the one complete backspace/delete key as label, bounds, confidence and fully_visible. Use null when absent, clipped, ambiguous, or confused with an App delete control. This is read-only geometry and never authorizes clearing by itself.
6. keyboard.literal_keys: the local, goal-derived whitelist is {json.dumps(literal_key_targets, ensure_ascii=False, separators=(',', ':'))}. Report only complete visible keys whose inserted value occurs in that exact whitelist, at most once per distinct value and at most eight total. Every literal-key object MUST contain exactly these six fields and never omit any of them: value, label, key_kind, bounds, confidence, fully_visible. When the whitelist is empty, literal_keys MUST be []. QWERTY alphabet letters and Chinese characters MUST NEVER be enumerated here, even when they occur in input_text, because qwerty_anchors and the verified pinyin transaction already represent them. Never enumerate a keyboard row. For a whitelisted space use value=" " and key_kind="space". For every other whitelisted key use key_kind="character" and require label to equal value literally. The large central PRIMARY glyph of the whole directly tappable key MUST equal value. A small corner glyph, superscript digit, alternate symbol, swipe hint or long-press hint printed on an alphabet key is NOT a literal key and MUST NEVER be reported here. If the whitelisted value exists only as such a secondary hint, leave literal_keys empty and report a separately visible direction-explicit numeric/symbol layout switch instead. Bounds must enclose the whole direct key, never only the secondary glyph. Never include backspace, enter, send/search, emoji, voice, assistant, shift, or layout switches.
7. keyboard.enter_key: report at most one complete visible keyboard action key using exactly label, bounds, confidence, fully_visible and key_action. key_action must be one of newline, send, search, done, next, unknown and must describe the key's current visible behavior, never the requested goal. A plain multiline Return/Enter key may be newline. A key visibly labelled or iconographically acting as Send/Search/Done/Next must use that action and can never authorize a newline. The current transaction needs a newline={str(enter_required).lower()} and multiline={str(active_multiline).lower()}, but those facts do not change the visual classification.
8. keyboard.layout_switches: enumerate only compact visible keys with an explicit destination layout: qwerty, numeric, or symbol. Copy the literal label and report current_layout and target_layout; never infer a destination from the goal alone.
7. keyboard.case_mode and keyboard.case_switch apply only to direct_latin QWERTY. case_mode is lower, upper, or unknown from the visible letter glyphs. case_switch is null unless a complete visible shift/case key and its lower↔upper direction are independently clear.
Determine keyboard.input_mode only from the current whole keyboard image, never from the goal, the JSON example, or the mode-switch key label alone. Visible Chinese composition/candidates or pinyin separators prove chinese_pinyin. A plain Latin QWERTY state with no Chinese composition/candidate strip may prove direct_latin only when the whole keyboard provides independent current-mode evidence. If the whole keyboard does not prove the current mode, use unknown and set mode_switch to null.
When a visible preedit composition itself exactly matches a complete visible candidate, that exact candidate MUST be enumerated with its own bounds. Omitting the exact candidate while reporting the matching preedit is an incomplete audit; never silently turn useful target text into a clear/delete instruction.
keyboard.mode_switch.current_mode MUST equal keyboard.input_mode whenever input_mode is known, and target_mode MUST be the other supported mode. Across real keyboards the visible key label may name either the current mode or the destination mode: for example, 英/EN can be shown while Chinese pinyin is current and pressing it enters direct Latin, or while direct Latin is current and pressing it enters Chinese. Copy the literal label, but never derive current_mode or target_mode from that label. If the direction is not independently clear from the whole keyboard state, set mode_switch to null.
For a text-entry verification goal, report the proven current keyboard.input_mode; keyboard.mode_switch is optional and should be null unless its direction is independently unambiguous. Never invent a switch direction merely because the goal asks for text entry.
keyboard.mode_switch MUST be either null or an object with exactly these five fields: label, bounds, confidence, current_mode, target_mode. Never omit confidence or target_mode. Valid non-null shapes in the two directions are:
{{"label":"英","bounds":[0,0,1000,1000],"confidence":0.0,"current_mode":"chinese_pinyin","target_mode":"direct_latin"}}
{{"label":"英","bounds":[0,0,1000,1000],"confidence":0.0,"current_mode":"direct_latin","target_mode":"chinese_pinyin"}}
These are shape examples only. Copy the literal visible label and measured bounds from Image 1, set confidence from the visible evidence, and choose the direction from the independently proven current keyboard state. Never copy either example merely to satisfy the goal.
keyboard.case_switch uses the same five field names, but current_mode and target_mode are lower or upper. It is valid only for direct_latin QWERTY and a visible shift/case glyph. Example shape: {{"label":"⇧","bounds":[0,0,1000,1000],"confidence":0.0,"current_mode":"lower","target_mode":"upper"}}.
Do not plan, suggest, authorize, or perform any action.
{coordinate_contract}
Use text="" for a visibly empty application field. Copy placeholders and visible_editable_cues literally; do not infer them from the goal. caret_line_index is the zero-based VISUAL row containing the complete visible insertion caret, or null when the caret row is absent, clipped, or ambiguous. It is a read-only geometry fact and by itself never proves a user-entered newline. right_button describes a trailing utility control; it is structural evidence only and is never authorized for activation. Set it to null when no separate trailing control is visible.
An automatic visual line wrap inside a narrow editable field is presentation only: join the continuous visible glyph sequence and do not insert "\\n" into text. Report a newline character only when the image independently proves an actual user-entered line break; if that distinction is not visually provable, do not invent a newline from row layout alone.
Return exactly this JSON schema and no other fields. Emit one compact minified
JSON object on a single line, without Markdown or explanatory whitespace:
{{"protocol_version":"{INPUT_STRUCTURE_AUDIT_VERSION}",
"application_inputs":[{{"structure_id":"app-input-1","bounds":[0,0,1000,1000],
"fully_visible":true,"text":"","placeholder":"visible placeholder or empty","field_labels":["literal visible field label"],
"visible_editable_cues":["literal visible cue"],"caret_line_index":null,"confidence":0.0,
"right_button":null}}],
"ime_preedit_regions":[{{"region_id":"ime-preedit-1","bounds":[0,0,1000,1000],
"text":"visible composition text or empty","confidence":0.0,
"candidates":[{{"text":"literal candidate","bounds":[0,0,1000,1000],"confidence":0.0,"fully_visible":true}}]}}],
"keyboard":{{"visible":true,"bounds":[0,0,1000,1000],"layout":"qwerty",
"input_mode":"unknown","case_mode":"unknown","qwerty_anchors":{{"q":[0,0],"p":[0,0],"a":[0,0],"l":[0,0],"z":[0,0],"m":[0,0],"backspace":[0,0]}},"mode_switch":null,
"backspace_key":{{"label":"⌫","bounds":[0,0,1000,1000],"confidence":0.0,"fully_visible":true}},
"enter_key":{{"label":"↵","bounds":[0,0,1000,1000],"confidence":0.0,"fully_visible":true,"key_action":"newline"}},
"case_switch":null,"literal_keys":{literal_keys_example_json},
"layout_switches":[{{"label":"123","bounds":[0,0,1000,1000],"confidence":0.0,"current_layout":"qwerty","target_layout":"numeric"}}]}}}}
When no keyboard is visible, keyboard must be {{"visible":false,"bounds":null,"layout":"unknown","input_mode":"unknown","case_mode":"unknown","qwerty_anchors":null,"mode_switch":null,"backspace_key":null,"enter_key":null,"case_switch":null,"literal_keys":[],"layout_switches":[]}}.
Return empty arrays when their geometry is not visible. Never merge a clipped structure with a complete structure, and never copy an IME pre-edit region into application_inputs.
"""


def _input_audit_detail_note(
    roi_bounds: tuple[int, int, int, int] | None,
) -> str:
    if roi_bounds is None:
        return "No detail crop is provided."
    return (
        f"Image 2 is only a magnified read-only crop of Image 1 at {list(roi_bounds)}. "
        "Use it to read details, but never use Image 2 as a coordinate system."
    )


def _map_input_structure_crop_audit_to_full(
    raw: str,
    *,
    roi_bounds: tuple[int, int, int, int],
) -> str:
    """Map strict crop-local input facts into the full-frame 0..1000 space."""

    payload = _extract_json_object(raw)
    left, top, right, bottom = roi_bounds

    def map_bounds(value: Any, field_name: str) -> list[int]:
        if not _valid_1000_bounds(value):
            raise VisionAgentError(f"{field_name} 不是有效的 crop-local bounds。")
        local = [float(part) for part in value]
        internal_edges = (
            (left > 0 and local[0] < 15)
            or (top > 0 and local[1] < 15)
            or (right < 1000 and local[2] > 985)
            or (bottom < 1000 and local[3] > 985)
        )
        if internal_edges:
            raise VisionAgentError(f"{field_name} 接触 crop 内部边界，不能证明完整可见。")
        return [
            round(left + local[0] * (right - left) / 1000.0),
            round(top + local[1] * (bottom - top) / 1000.0),
            round(left + local[2] * (right - left) / 1000.0),
            round(top + local[3] * (bottom - top) / 1000.0),
        ]

    def map_point(value: Any, field_name: str) -> list[int]:
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(
                isinstance(part, bool) or not isinstance(part, (int, float))
                for part in value
            )
            or not all(0 <= float(part) <= 1000 for part in value)
        ):
            raise VisionAgentError(f"{field_name} 不是有效的 crop-local point。")
        return [
            round(left + float(value[0]) * (right - left) / 1000.0),
            round(top + float(value[1]) * (bottom - top) / 1000.0),
        ]

    application_inputs = payload.get("application_inputs")
    if isinstance(application_inputs, list):
        for index, item in enumerate(application_inputs):
            if not isinstance(item, dict):
                continue
            if "bounds" in item:
                item["bounds"] = map_bounds(
                    item["bounds"],
                    f"application_inputs[{index}].bounds",
                )
            right_button = item.get("right_button")
            if isinstance(right_button, dict) and "bounds" in right_button:
                right_button["bounds"] = map_bounds(
                    right_button["bounds"],
                    f"application_inputs[{index}].right_button.bounds",
                )

    ime_regions = payload.get("ime_preedit_regions")
    if isinstance(ime_regions, list):
        for index, item in enumerate(ime_regions):
            if isinstance(item, dict) and "bounds" in item:
                item["bounds"] = map_bounds(
                    item["bounds"],
                    f"ime_preedit_regions[{index}].bounds",
                )
                candidates = item.get("candidates")
                if isinstance(candidates, list):
                    for candidate_index, candidate in enumerate(candidates):
                        if isinstance(candidate, dict) and "bounds" in candidate:
                            candidate["bounds"] = map_bounds(
                                candidate["bounds"],
                                "ime_preedit_regions"
                                f"[{index}].candidates[{candidate_index}].bounds",
                            )

    keyboard = payload.get("keyboard")
    if isinstance(keyboard, dict):
        if keyboard.get("bounds") is not None:
            keyboard["bounds"] = map_bounds(
                keyboard["bounds"],
                "keyboard.bounds",
            )
        anchors = keyboard.get("qwerty_anchors")
        if isinstance(anchors, dict):
            for key, value in list(anchors.items()):
                anchors[key] = map_point(value, f"keyboard.qwerty_anchors.{key}")
        mode_switch = keyboard.get("mode_switch")
        if isinstance(mode_switch, dict) and "bounds" in mode_switch:
            mode_switch["bounds"] = map_bounds(
                mode_switch["bounds"],
                "keyboard.mode_switch.bounds",
            )
        backspace_key = keyboard.get("backspace_key")
        if isinstance(backspace_key, dict) and "bounds" in backspace_key:
            backspace_key["bounds"] = map_bounds(
                backspace_key["bounds"],
                "keyboard.backspace_key.bounds",
            )
        enter_key = keyboard.get("enter_key")
        if isinstance(enter_key, dict) and "bounds" in enter_key:
            enter_key["bounds"] = map_bounds(
                enter_key["bounds"],
                "keyboard.enter_key.bounds",
            )
        case_switch = keyboard.get("case_switch")
        if isinstance(case_switch, dict) and "bounds" in case_switch:
            case_switch["bounds"] = map_bounds(
                case_switch["bounds"],
                "keyboard.case_switch.bounds",
            )
        for collection_name in ("literal_keys", "layout_switches"):
            collection = keyboard.get(collection_name)
            if isinstance(collection, list):
                for index, item in enumerate(collection):
                    if isinstance(item, dict) and "bounds" in item:
                        item["bounds"] = map_bounds(
                            item["bounds"],
                            f"keyboard.{collection_name}[{index}].bounds",
                        )

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _normalize_input_audit_pixel_coordinates(
    payload: dict[str, Any],
    *,
    frame_size: tuple[int, int],
) -> None:
    """Normalize a whole-frame audit emitted in source-image pixel space.

    The formal contract is 0..1000 on both axes, but a model can consistently
    copy the actual source-image coordinates instead. Accept that alternative
    only when at least one coordinate exceeds 1000 and every declared geometry
    value fits the current frame exactly; then convert the entire audit as one
    coordinate system. Mixed or out-of-frame geometry remains invalid.
    """

    frame_width, frame_height = frame_size
    if frame_width <= 0 or frame_height <= 0:
        return
    bounds_refs: list[tuple[dict[str, Any], str]] = []
    point_refs: list[tuple[dict[str, Any], str]] = []

    def add_bounds(owner: Any, key: str = "bounds") -> None:
        if isinstance(owner, dict) and owner.get(key) is not None:
            bounds_refs.append((owner, key))

    def add_point(owner: Any, key: str) -> None:
        if isinstance(owner, dict) and owner.get(key) is not None:
            point_refs.append((owner, key))

    for item in payload.get("application_inputs") or []:
        add_bounds(item)
        if isinstance(item, dict):
            add_bounds(item.get("right_button"))
    for region in payload.get("ime_preedit_regions") or []:
        add_bounds(region)
        if isinstance(region, dict):
            for candidate in region.get("candidates") or []:
                add_bounds(candidate)
    keyboard = payload.get("keyboard")
    if isinstance(keyboard, dict):
        add_bounds(keyboard)
        for key in ("mode_switch", "backspace_key", "enter_key", "case_switch"):
            add_bounds(keyboard.get(key))
        for collection_name in ("literal_keys", "layout_switches"):
            for item in keyboard.get(collection_name) or []:
                add_bounds(item)
        anchors = keyboard.get("qwerty_anchors")
        if isinstance(anchors, dict):
            for key in anchors:
                add_point(anchors, key)

    coordinates: list[tuple[float, float]] = []
    for owner, key in bounds_refs:
        value = owner.get(key)
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 4
            or any(
                isinstance(part, bool) or not isinstance(part, (int, float))
                for part in value
            )
        ):
            return
        left, top, right, bottom = (float(part) for part in value)
        if not (0 <= left < right <= frame_width and 0 <= top < bottom <= frame_height):
            return
        coordinates.extend(((left, top), (right, bottom)))
    for owner, key in point_refs:
        value = owner.get(key)
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(
                isinstance(part, bool) or not isinstance(part, (int, float))
                for part in value
            )
        ):
            return
        x, y = (float(part) for part in value)
        if not (0 <= x <= frame_width and 0 <= y <= frame_height):
            return
        coordinates.append((x, y))
    if not coordinates or not any(x > 1000 or y > 1000 for x, y in coordinates):
        return

    for owner, key in bounds_refs:
        left, top, right, bottom = (float(part) for part in owner[key])
        owner[key] = [
            round(left * 1000 / frame_width),
            round(top * 1000 / frame_height),
            round(right * 1000 / frame_width),
            round(bottom * 1000 / frame_height),
        ]
    for owner, key in point_refs:
        x, y = (float(part) for part in owner[key])
        owner[key] = [
            round(x * 1000 / frame_width),
            round(y * 1000 / frame_height),
        ]


def _normalize_input_audit_portrait_grid_from_local_rows(
    payload: dict[str, Any],
    *,
    locally_snapped_anchors: dict[str, list[int]],
) -> None:
    """Normalize a model portrait Y grid from independently snapped rows.

    Some audits keep X in 0..1000 but emit Y in a taller logical phone grid.
    No conventional screen height is guessed.  A single affine Y transform is
    accepted only when all three model QWERTY rows map to the stable local OCR
    rows with low residual, then it is applied atomically to the whole audit.
    """

    keyboard = payload.get("keyboard")
    raw_anchors = keyboard.get("qwerty_anchors") if isinstance(keyboard, dict) else None
    if not isinstance(raw_anchors, dict):
        return
    try:
        raw_rows = (
            statistics.mean((float(raw_anchors["q"][1]), float(raw_anchors["p"][1]))),
            statistics.mean((float(raw_anchors["a"][1]), float(raw_anchors["l"][1]))),
            statistics.mean(
                (
                    float(raw_anchors["z"][1]),
                    float(raw_anchors["m"][1]),
                    float(raw_anchors["backspace"][1]),
                )
            ),
        )
        local_rows = (
            statistics.mean(
                (
                    float(locally_snapped_anchors["q"][1]),
                    float(locally_snapped_anchors["p"][1]),
                )
            ),
            statistics.mean(
                (
                    float(locally_snapped_anchors["a"][1]),
                    float(locally_snapped_anchors["l"][1]),
                )
            ),
            statistics.mean(
                (
                    float(locally_snapped_anchors["z"][1]),
                    float(locally_snapped_anchors["m"][1]),
                    float(locally_snapped_anchors["backspace"][1]),
                )
            ),
        )
    except (KeyError, TypeError, ValueError):
        return
    if not (
        raw_rows[2] > 1000
        and raw_rows[0] < raw_rows[1] < raw_rows[2]
        and local_rows[0] < local_rows[1] < local_rows[2]
    ):
        return
    raw_mean = statistics.mean(raw_rows)
    local_mean = statistics.mean(local_rows)
    denominator = sum((value - raw_mean) ** 2 for value in raw_rows)
    if denominator <= 0:
        return
    scale = sum(
        (raw_value - raw_mean) * (local_value - local_mean)
        for raw_value, local_value in zip(raw_rows, local_rows)
    ) / denominator
    offset = local_mean - scale * raw_mean
    if not 0.25 <= scale <= 1.0 or max(
        abs(scale * raw_value + offset - local_value)
        for raw_value, local_value in zip(raw_rows, local_rows)
    ) > 15:
        return

    bounds_refs: list[tuple[dict[str, Any], str]] = []
    point_refs: list[tuple[dict[str, Any], str]] = []

    def add_bounds(owner: Any, key: str = "bounds") -> None:
        if isinstance(owner, dict) and owner.get(key) is not None:
            bounds_refs.append((owner, key))

    for item in payload.get("application_inputs") or []:
        add_bounds(item)
        if isinstance(item, dict):
            add_bounds(item.get("right_button"))
    for region in payload.get("ime_preedit_regions") or []:
        add_bounds(region)
        if isinstance(region, dict):
            for candidate in region.get("candidates") or []:
                add_bounds(candidate)
    add_bounds(keyboard)
    for key in ("mode_switch", "backspace_key", "enter_key", "case_switch"):
        add_bounds(keyboard.get(key))
    for collection_name in ("literal_keys", "layout_switches"):
        for item in keyboard.get(collection_name) or []:
            add_bounds(item)
    for key in raw_anchors:
        point_refs.append((raw_anchors, key))

    transformed_bounds: list[tuple[dict[str, Any], str, list[int]]] = []
    transformed_points: list[tuple[dict[str, Any], str, list[int]]] = []
    for owner, key in bounds_refs:
        value = owner.get(key)
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 4
            or any(
                isinstance(part, bool) or not isinstance(part, (int, float))
                for part in value
            )
        ):
            return
        left, top, right, bottom = (float(part) for part in value)
        normalized_top = scale * top + offset
        normalized_bottom = scale * bottom + offset
        if not (
            0 <= left < right <= 1000
            and 0 <= normalized_top < normalized_bottom <= 1000
        ):
            return
        transformed_bounds.append(
            (
                owner,
                key,
                [round(left), round(normalized_top), round(right), round(normalized_bottom)],
            )
        )
    for owner, key in point_refs:
        value = owner.get(key)
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(
                isinstance(part, bool) or not isinstance(part, (int, float))
                for part in value
            )
        ):
            return
        x, y = (float(part) for part in value)
        normalized_y = scale * y + offset
        if not (0 <= x <= 1000 and 0 <= normalized_y <= 1000):
            return
        transformed_points.append(
            (owner, key, [round(x), round(normalized_y)])
        )
    for owner, key, value in transformed_bounds:
        owner[key] = value
    for owner, key, value in transformed_points:
        owner[key] = value


def _reattach_input_audit_to_unique_scene_field(
    scene: UIScene,
    application_inputs: Any,
    *,
    goal_context: dict[str, Any],
) -> None:
    """Use one same-frame typed field as local input geometry authority.

    The compact scene contributes only a coarse rectangle from the same
    fingerprint.  The dedicated audit must still independently provide the
    editable cues, exact value and field label.  Ambiguous or conflicting
    fields are never reattached.
    """

    if not isinstance(application_inputs, list) or not application_inputs:
        return
    active_field_id, active_field_label, _active_multiline = (
        _goal_active_input_field(goal_context)
    )
    audit_matches = []
    for item in application_inputs:
        if not isinstance(item, dict):
            continue
        labels = item.get("field_labels")
        if not isinstance(labels, list):
            continue
        if active_field_label:
            if sum(
                isinstance(label, str)
                and label.strip().casefold() == active_field_label.casefold()
                for label in labels
            ) != 1:
                continue
        audit_matches.append(item)
    if len(audit_matches) != 1:
        return

    scene_matches = []
    for element in scene.elements:
        if (
            element.role != "input"
            or element.confidence < 0.9
            or element.states.get("fully_visible") is not True
            or element.states.get("goal_relevant") is not True
        ):
            continue
        visible_field_id = str(element.states.get("input_field_id") or "").strip()
        if (
            active_field_id
            and visible_field_id
            and visible_field_id != active_field_id
        ):
            continue
        visible_identity = {
            str(element.label or "").strip().casefold(),
            str(element.states.get("placeholder") or "").strip().casefold(),
            *(
                str(value).strip().casefold()
                for value in element.evidence
                if str(value).strip()
            ),
        }
        if active_field_label and active_field_label.casefold() not in visible_identity:
            continue
        scene_matches.append(element)
    if len(scene_matches) != 1:
        return
    bounds = scene_matches[0].bounds
    if (
        not isinstance(bounds, (list, tuple))
        or len(bounds) != 4
        or not all(
            isinstance(part, (int, float)) and not isinstance(part, bool)
            for part in bounds
        )
    ):
        return
    normalized = [round(float(part) * 1000) for part in bounds]
    if not _valid_1000_bounds(normalized):
        return
    audit_matches[0]["bounds"] = normalized


_JSON_STRUCTURAL_PUNCTUATION = "{}[],:"
_JSON_STRUCTURAL_REPAIR_WINDOW = 2
_MAX_JSON_STRUCTURAL_REPAIR_CANDIDATES = 40
_MAX_JSON_STRUCTURAL_REPAIR_CHARS = 16000


class _DuplicateJSONKeyError(ValueError):
    pass


def _reject_duplicate_json_object_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKeyError(key)
        result[key] = value
    return result


def _load_json_without_duplicate_keys(raw: str) -> Any:
    return json.loads(raw, object_pairs_hook=_reject_duplicate_json_object_pairs)


def _strict_foreground_app_identity_audit(
    raw: str,
) -> tuple[str, float, tuple[str, ...]]:
    text = str(raw or "").strip()
    if not text or text.startswith("```"):
        raise VisionAgentError("前台应用身份审计必须返回纯 JSON 对象。")
    try:
        payload = _load_json_without_duplicate_keys(text)
    except _DuplicateJSONKeyError as exc:
        raise VisionAgentError(
            f"前台应用身份审计包含重复 JSON 字段：{exc}"
        ) from exc
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise VisionAgentError(f"前台应用身份审计 JSON 无法解析：{exc}") from exc
    if not isinstance(payload, dict):
        raise VisionAgentError("前台应用身份审计必须是 JSON 对象。")
    required = {
        "protocol_version",
        "foreground_app_id",
        "confidence",
        "evidence",
    }
    missing = required - set(payload)
    unexpected = set(payload) - required
    if missing:
        raise VisionAgentError(
            "前台应用身份审计缺少字段：" + ", ".join(sorted(missing))
        )
    if unexpected:
        raise VisionAgentError(
            "前台应用身份审计包含协议外字段："
            + ", ".join(sorted(map(str, unexpected)))
        )
    if payload["protocol_version"] != FOREGROUND_APP_IDENTITY_AUDIT_VERSION:
        raise VisionAgentError("前台应用身份审计协议版本不匹配。")
    app_id = str(payload["foreground_app_id"] or "").strip().casefold()
    if not _FOREGROUND_APP_ID_PATTERN.fullmatch(app_id):
        raise VisionAgentError("前台应用身份审计 app_id 格式无效。")
    if _is_foreground_app_identity_placeholder(app_id):
        raise VisionAgentError("前台应用身份审计不得返回引用占位符。")
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise VisionAgentError("前台应用身份审计 confidence 格式无效。")
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise VisionAgentError("前台应用身份审计 confidence 越界。")
    evidence = payload["evidence"]
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= 3:
        raise VisionAgentError("前台应用身份审计 evidence 不安全或格式无效。")
    evidence_items = tuple(
        str(item).strip()
        for item in evidence
        if isinstance(item, str) and item.strip() and len(item.strip()) <= 120
    )
    if len(evidence_items) != len(evidence):
        raise VisionAgentError("前台应用身份审计 evidence 不安全或格式无效。")
    identity_established = (
        app_id != "unknown"
        and confidence >= _MIN_FOREGROUND_APP_IDENTITY_CONFIDENCE
    )
    if not identity_established:
        app_id = "unknown"
        evidence_items = tuple(
            item
            for item in evidence_items
            if camera_alignment_evidence_is_safe(item)
        )
    elif any(
        not camera_alignment_evidence_is_safe(item) for item in evidence_items
    ):
        raise VisionAgentError("前台应用身份审计 evidence 不安全或格式无效。")
    return app_id, confidence, evidence_items


def _extract_compact_json_object(raw: str) -> dict[str, Any]:
    """Extract only compact-scene JSON while rejecting duplicate keys."""

    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = _load_json_without_duplicate_keys(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise VisionAgentError("模型没有返回 JSON 对象。")
        try:
            value = _load_json_without_duplicate_keys(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise VisionAgentError(f"模型返回的 JSON 无法解析：{exc}") from exc
        except _DuplicateJSONKeyError as exc:
            raise VisionAgentError(
                f"compact 响应包含重复 JSON 字段：{exc}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise VisionAgentError(f"模型返回的 JSON 无法解析：{exc}") from exc
    except _DuplicateJSONKeyError as exc:
        raise VisionAgentError(
            f"compact 响应包含重复 JSON 字段：{exc}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise VisionAgentError(f"模型返回的 JSON 无法解析：{exc}") from exc
    if not isinstance(value, dict):
        raise VisionAgentError("模型返回值必须是 JSON 对象。")
    return value


def _compact_response_has_repairable_syntax_error(raw: str) -> bool:
    try:
        _extract_compact_json_object(raw)
    except VisionAgentError as exc:
        cause = exc.__cause__
        return cause is None or isinstance(cause, json.JSONDecodeError)
    return False


def _targeted_response_has_repairable_syntax_error(raw: str) -> bool:
    """Return true only when the targeted response itself is invalid JSON.

    A schema, enum, evidence, or geometry rejection is not a punctuation error
    and must retain its original fail-closed diagnostic. Trying structural edits
    on valid JSON can otherwise hide the actual protocol violation behind the
    misleading message that no unique punctuation repair exists.
    """

    try:
        _extract_targeted_delta_json_object(raw)
    except VisionAgentError as exc:
        return isinstance(exc.__cause__, json.JSONDecodeError)
    return False


def _single_json_structural_edits(raw: str) -> Iterator[str]:
    """Yield bounded one-character edits around the original parser error."""

    text = str(raw or "").strip()
    if (
        not text.startswith("{")
        or len(text) > _MAX_JSON_STRUCTURAL_REPAIR_CHARS
    ):
        return
    try:
        _load_json_without_duplicate_keys(text)
    except json.JSONDecodeError as error:
        error_position = error.pos
    except (TypeError, ValueError):
        return
    else:
        return
    if not text.endswith("}") and error_position != len(text):
        return

    boundaries: list[bool] = [True]
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        boundaries.append(not in_string)

    start = max(0, error_position - _JSON_STRUCTURAL_REPAIR_WINDOW)
    stop = min(len(text), error_position + _JSON_STRUCTURAL_REPAIR_WINDOW + 1)
    seen: set[str] = set()
    yielded = 0
    for index in range(start, stop):
        character = text[index]
        if character not in _JSON_STRUCTURAL_PUNCTUATION or not boundaries[index]:
            continue
        candidate = text[:index] + text[index + 1 :]
        if candidate not in seen:
            seen.add(candidate)
            yield candidate
            yielded += 1
            if yielded >= _MAX_JSON_STRUCTURAL_REPAIR_CANDIDATES:
                return

    insertion_stop = min(
        len(text),
        error_position + _JSON_STRUCTURAL_REPAIR_WINDOW,
    )
    for index in range(start, insertion_stop + 1):
        if not boundaries[index]:
            continue
        for character in _JSON_STRUCTURAL_PUNCTUATION:
            candidate = text[:index] + character + text[index:]
            if candidate not in seen:
                seen.add(candidate)
                yield candidate
                yielded += 1
                if yielded >= _MAX_JSON_STRUCTURAL_REPAIR_CANDIDATES:
                    return


def _parse_scene_after_unique_structural_edit(
    raw: str,
    *,
    fingerprint: str,
    goal_context: dict[str, Any] | None = None,
    allow_invalid_system_ui_unknown: bool = False,
    camera_layout_orientation: str | None = None,
    camera_alignment_override: CameraAlignmentFacts | None = None,
) -> UIScene | None:
    """Accept one punctuation edit only when exactly one strict scene survives."""

    result = _unique_strict_structural_scene_edit(
        raw,
        fingerprint=fingerprint,
        goal_context=goal_context,
        allow_invalid_system_ui_unknown=allow_invalid_system_ui_unknown,
        camera_layout_orientation=camera_layout_orientation,
        camera_alignment_override=camera_alignment_override,
    )
    return result[1] if result is not None else None


def _parse_targeted_delta_after_unique_structural_edit(
    raw: str,
    *,
    base_scene: UIScene,
    fingerprint: str,
    goal_context: dict[str, Any] | None = None,
) -> UIScene | None:
    """Accept one punctuation edit only for one strict targeted delta."""

    accepted: list[UIScene] = []
    for candidate in _single_json_structural_edits(raw):
        try:
            decoded = _load_json_without_duplicate_keys(candidate)
            _normalize_targeted_delta_evidence_shorthand(decoded)
            if not _matches_targeted_delta_schema(decoded):
                continue
            scene = _parse_targeted_scene_delta(
                candidate,
                base_scene=base_scene,
                fingerprint=fingerprint,
                goal_context=goal_context,
            )
        except (json.JSONDecodeError, ValueError, VisionAgentError):
            continue
        accepted.append(scene)
        if len(accepted) > 1:
            return None
    return accepted[0] if accepted else None


def _unique_strict_structural_scene_edit(
    raw: str,
    *,
    fingerprint: str,
    goal_context: dict[str, Any] | None = None,
    allow_invalid_system_ui_unknown: bool = False,
    camera_layout_orientation: str | None = None,
    camera_alignment_override: CameraAlignmentFacts | None = None,
) -> tuple[str, UIScene] | None:
    accepted: list[tuple[str, UIScene]] = []
    for candidate in _single_json_structural_edits(raw):
        try:
            decoded = _load_json_without_duplicate_keys(candidate)
            if not _matches_compact_repair_schema(decoded):
                continue
            scene = _parse_scene(
                candidate,
                fingerprint=fingerprint,
                goal_context=goal_context,
                allow_invalid_system_ui_unknown=allow_invalid_system_ui_unknown,
                camera_layout_orientation=camera_layout_orientation,
                camera_alignment_override=camera_alignment_override,
            )
        except (json.JSONDecodeError, ValueError, VisionAgentError):
            continue
        accepted.append((candidate, scene))
        if len(accepted) > 1:
            return None
    return accepted[0] if accepted else None


def _matches_compact_repair_schema(payload: Any) -> bool:
    """Require the exact compact container shape before normalizers run."""

    if not isinstance(payload, dict):
        return False
    required = {
        "protocol_version",
        "screen_id",
        "summary",
        "system_ui",
        "camera_alignment",
        "elements",
        "overlays",
        "stable",
        "confidence",
        "fingerprint",
    }
    allowed = required | {"foreground_app_id", "app_id"}
    if not required.issubset(payload) or not set(payload).issubset(allowed):
        return False
    if not ({"foreground_app_id", "app_id"} & set(payload)):
        return False
    system_ui = payload.get("system_ui")
    alignment = payload.get("camera_alignment")
    elements = payload.get("elements")
    if (
        payload.get("protocol_version") != UI_SCENE_PROTOCOL_VERSION
        or not isinstance(system_ui, dict)
        or set(system_ui)
        != {"immersive_or_fullscreen", "navigation_bar_visible"}
        or not isinstance(alignment, dict)
        or set(alignment)
        != {
            "camera_layout_orientation",
            "phone_content_rotation",
            "confidence",
            "evidence",
        }
        or not isinstance(elements, list)
        or len(elements) > MAX_COMPACT_ELEMENTS
        or not isinstance(payload.get("overlays"), list)
    ):
        return False
    element_fields = {
        "element_id",
        "role",
        "meaning",
        "label",
        "bounds",
        "confidence",
        "states",
        "evidence",
    }
    for element in elements:
        if not isinstance(element, dict) or set(element) != element_fields:
            return False
        if (
            not isinstance(element["bounds"], list)
            or len(element["bounds"]) != 4
            or not isinstance(element["states"], dict)
            or not isinstance(element["evidence"], list)
        ):
            return False
    return True


def _matches_targeted_delta_schema(payload: Any) -> bool:
    """Require the exact small refinement shape before any scene merge."""

    if not isinstance(payload, dict):
        return False
    if set(payload) != {
        "protocol_version",
        "summary_addendum",
        "elements",
        "confidence",
    }:
        return False
    if payload.get("protocol_version") != TARGETED_SCENE_DELTA_PROTOCOL_VERSION:
        return False
    summary_addendum = payload.get("summary_addendum")
    elements = payload.get("elements")
    confidence = payload.get("confidence")
    if (
        not isinstance(summary_addendum, str)
        or len(summary_addendum.strip()) > 120
        or not isinstance(elements, list)
        or len(elements) > MAX_COMPACT_ELEMENTS
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        return False
    element_fields = {
        "element_id",
        "role",
        "meaning",
        "label",
        "bounds",
        "confidence",
        "states",
        "evidence",
    }
    return all(
        isinstance(element, dict)
        and set(element) == element_fields
        and isinstance(element.get("bounds"), list)
        and len(element["bounds"]) == 4
        and isinstance(element.get("states"), dict)
        and isinstance(element.get("evidence"), list)
        for element in elements
    )


def _normalize_targeted_delta_evidence_shorthand(payload: Any) -> None:
    """Normalize only the existing one-string evidence JSON shorthand."""

    if not isinstance(payload, dict):
        return
    elements = payload.get("elements")
    if not isinstance(elements, list):
        return
    for element in elements:
        if not isinstance(element, dict):
            continue
        evidence = element.get("evidence")
        if isinstance(evidence, str):
            text = evidence.strip()
            element["evidence"] = [text] if text else []


def _normalize_targeted_delta_xywh_bounds_shorthand(payload: Any) -> None:
    """Normalize only exact finite rectangular bounds shorthands."""

    if not isinstance(payload, dict) or not isinstance(payload.get("elements"), list):
        return
    for element in payload["elements"]:
        if not isinstance(element, dict):
            continue
        bounds = element.get("bounds")
        if not isinstance(bounds, dict):
            continue
        if set(bounds) == {"x", "y", "w", "h"}:
            keys = ("x", "y", "w", "h")
        elif set(bounds) == {"x", "y", "width", "height"}:
            keys = ("x", "y", "width", "height")
        elif set(bounds) == {"x1", "y1", "x2", "y2"}:
            values = tuple(bounds[key] for key in ("x1", "y1", "x2", "y2"))
            if (
                any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    for value in values
                )
                or any(not math.isfinite(float(value)) for value in values)
            ):
                continue
            x1, y1, x2, y2 = (float(value) for value in values)
            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > 1000 or y2 > 1000:
                continue
            element["bounds"] = [x1, y1, x2, y2]
            continue
        else:
            continue
        values = tuple(bounds[key] for key in keys)
        if (
            any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values)
            or any(not math.isfinite(float(value)) for value in values)
        ):
            continue
        x, y, width, height = (float(value) for value in values)
        if (
            x < 0
            or y < 0
            or width <= 0
            or height <= 0
            or x + width > 1000
            or y + height > 1000
        ):
            continue
        element["bounds"] = [x, y, x + width, y + height]


def _reattach_invalid_targeted_xywh_to_unique_base_element(
    payload: Any,
    base_scene: UIScene,
) -> None:
    """Discard an unknown coordinate canvas only when identity is already exact.

    Some vision responses preserve the strict delta shape but emit ``x/y/w/h``
    on an undocumented portrait canvas whose y values exceed 1000.  Guessing a
    scale would turn model geometry into authority.  A unique element already
    observed on the same frame can instead keep its own rough bounds; the
    action adapter still requires independent crop geometry audits before any
    physical action.
    """

    if not isinstance(payload, dict) or not isinstance(payload.get("elements"), list):
        return
    for element in payload["elements"]:
        if not isinstance(element, dict):
            continue
        bounds = element.get("bounds")
        if not isinstance(bounds, dict) or set(bounds) not in (
            {"x", "y", "w", "h"},
            {"x", "y", "width", "height"},
        ):
            continue
        keys = (
            ("x", "y", "w", "h")
            if "w" in bounds
            else ("x", "y", "width", "height")
        )
        values = tuple(bounds[key] for key in keys)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            continue
        x, y, width, height = (float(value) for value in values)
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            continue
        # Valid normalized shorthand was already handled by the normalizer
        # above.  This path never interprets or rescales an unknown canvas.
        if x + width <= 1000 and y + height <= 1000:
            continue
        label = element.get("label")
        role = element.get("role")
        meaning = element.get("meaning")
        states = element.get("states")
        confidence = element.get("confidence")
        if (
            not isinstance(label, str)
            or not label.strip()
            or not isinstance(role, str)
            or not role.strip()
            or not isinstance(meaning, str)
            or not meaning.strip()
            or not isinstance(states, dict)
            or states.get("goal_relevant") is not True
            or states.get("fully_visible") is not True
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or float(confidence) < 0.9
        ):
            continue
        matches = tuple(
            candidate
            for candidate in base_scene.elements
            if candidate.label == label
            and candidate.role == role
            and candidate.meaning == meaning
            and candidate.confidence >= 0.9
            and candidate.states.get("goal_relevant") is True
            and candidate.states.get("fully_visible") is True
        )
        if len(matches) != 1:
            continue
        match = matches[0]
        element["element_id"] = match.element_id
        element["bounds"] = [part * 1000.0 for part in match.bounds]


def _extract_targeted_delta_json_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = _load_json_without_duplicate_keys(text)
    except json.JSONDecodeError as exc:
        raise VisionAgentError(
            f"目标精查返回的 JSON 无法解析：{exc}"
        ) from exc
    except _DuplicateJSONKeyError as exc:
        raise VisionAgentError(
            f"目标精查响应包含重复 JSON 字段：{exc}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise VisionAgentError(
            f"目标精查返回的 JSON 无法解析：{exc}"
        ) from exc
    if not isinstance(value, dict):
        raise VisionAgentError("目标精查返回值必须是 JSON 对象。")
    return value


def _parse_targeted_scene_delta(
    raw: str,
    *,
    base_scene: UIScene,
    fingerprint: str,
    goal_context: dict[str, Any] | None = None,
) -> UIScene:
    """Merge target-only evidence while preserving compact/local authority."""

    base_scene.validate()
    payload = _extract_targeted_delta_json_object(raw)
    _normalize_targeted_delta_evidence_shorthand(payload)
    _normalize_targeted_delta_xywh_bounds_shorthand(payload)
    _reattach_invalid_targeted_xywh_to_unique_base_element(payload, base_scene)
    _drop_out_of_range_non_goal_elements(payload, goal_context or {})
    if not _matches_targeted_delta_schema(payload):
        raise VisionAgentError(
            "目标精查结果不符合最小增量协议；只允许 protocol_version、"
            "summary_addendum、elements 和 confidence。"
        )

    addendum = payload["summary_addendum"].strip()
    summary = base_scene.summary
    if addendum and addendum not in summary:
        combined = f"{summary}；{addendum}" if summary else addendum
        if len(combined) <= 500:
            summary = combined
    full_payload = base_scene.to_dict()
    full_payload["summary"] = summary
    full_payload["elements"] = payload["elements"]
    full_payload["confidence"] = min(
        float(base_scene.confidence),
        float(payload["confidence"]),
    )
    parsed = _parse_scene(
        json.dumps(full_payload, ensure_ascii=False, separators=(",", ":")),
        fingerprint=fingerprint,
        goal_context=goal_context,
        allow_invalid_system_ui_unknown=True,
        camera_alignment_override=base_scene.camera_alignment,
    )
    merged = replace(
        parsed,
        app_id=base_scene.app_id,
        screen_id=base_scene.screen_id,
        summary=summary,
        system_ui=base_scene.system_ui,
        camera_alignment=base_scene.camera_alignment,
        overlays=base_scene.overlays,
        stable=base_scene.stable,
        fingerprint=fingerprint,
    )
    try:
        merged.validate()
    except (UISceneError, ValueError, TypeError) as exc:
        raise VisionAgentError(
            f"目标精查合并结果不符合场景协议：{exc}"
        ) from exc
    return merged


def _parse_scene(
    raw: str,
    *,
    fingerprint: str,
    goal_context: dict[str, Any] | None = None,
    allow_invalid_system_ui_unknown: bool = False,
    camera_layout_orientation: str | None = None,
    camera_alignment_override: CameraAlignmentFacts | None = None,
) -> UIScene:
    try:
        payload = _extract_json_object(raw)
        if "system_ui" not in payload:
            raise UISceneError(
                "新观察必须显式返回 scene.system_ui；无法判断时两项都写 unknown。"
            )
        if allow_invalid_system_ui_unknown:
            _fail_closed_invalid_system_ui(payload)
        _drop_forbidden_camera_alignment_evidence(payload)
        if camera_alignment_override is None:
            if "camera_alignment" not in payload:
                raise UISceneError(
                    "新观察必须显式返回 scene.camera_alignment。"
                )
            alignment = CameraAlignmentFacts.from_dict(
                payload["camera_alignment"]
            )
            if (
                camera_layout_orientation is not None
                and alignment.camera_layout_orientation
                != camera_layout_orientation
            ):
                raise UISceneError(
                    "模型报告的相机画布方向与本地稳定帧尺寸不一致。"
                )
        else:
            camera_alignment_override.validate()
            payload["camera_alignment"] = camera_alignment_override.to_dict()
        _normalize_compact_scene_payload(payload)
        _normalize_known_scene_enums(payload)
        _strip_model_authored_local_attestations(payload)
        _normalize_non_target_keyboard_switch(payload, goal_context or {})
        _normalize_reload_goal_safety(payload, goal_context or {})
        _strip_preliminary_elements_for_keyboard_mode_audit(
            payload,
            goal_context or {},
        )
        _normalize_tab_navigation_safety(payload, goal_context or {})
        _normalize_prefilled_input_structure(payload, goal_context or {})
        _normalize_local_text_clear_structure(payload, goal_context or {})
        _normalize_exact_target_ui_label_relevance(payload, goal_context or {})
        _normalize_page_title_identity(payload, goal_context or {})
        _normalize_unique_input_focus(payload)
        _drop_out_of_range_non_goal_elements(payload, goal_context or {})
        return UIScene.from_dict(
            payload,
            coordinate_scale=1000.0,
            stable_override=True,
            fingerprint_override=fingerprint,
        )
    except (UISceneError, ValueError, TypeError) as exc:
        raise VisionAgentError(f"通用页面观察结果不符合协议：{exc}") from exc


def _camera_layout_orientation(frame: Image.Image) -> str:
    if frame.width > frame.height:
        return "landscape"
    if frame.height > frame.width:
        return "portrait"
    return "square"


def _drop_forbidden_camera_alignment_evidence(payload: dict[str, Any]) -> None:
    """Remove peripheral HUD text without minting alignment authority from it."""

    alignment = payload.get("camera_alignment")
    if not isinstance(alignment, dict):
        return
    evidence = alignment.get("evidence")
    if not isinstance(evidence, list) or not all(
        isinstance(item, str) for item in evidence
    ):
        return
    retained = [
        item for item in evidence if camera_alignment_evidence_is_safe(item)
    ]
    if retained and len(retained) != len(evidence):
        alignment["evidence"] = retained
        return
    if not evidence or retained:
        return
    required = {
        "camera_layout_orientation",
        "phone_content_rotation",
        "confidence",
        "evidence",
    }
    confidence = alignment.get("confidence")
    if (
        set(alignment) != required
        or alignment.get("camera_layout_orientation")
        not in CAMERA_LAYOUT_ORIENTATIONS
        or alignment.get("phone_content_rotation") not in PHONE_CONTENT_ROTATIONS
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        # Do not let the repair hide malformed scalar fields.  The strict scene
        # parser below remains responsible for rejecting those payloads.
        return
    # Every evidence item came from coordinates or controller HUD text.  Such
    # text cannot prove phone rotation, but it also says nothing about the rest
    # of the page.  Downgrade only the unproved fact; later physical actions
    # still require their independent orientation credential.
    alignment["phone_content_rotation"] = "unknown"
    alignment["confidence"] = 0.0
    alignment["evidence"] = []


def _fail_closed_invalid_system_ui(payload: dict[str, Any]) -> None:
    """Replace only malformed fact values with two unknowns for a later audit.

    The container must still use the exact scene protocol shape. Missing or
    extra fields remain hard errors, and strings such as ``"true"`` are never
    coerced into booleans.
    """

    value = payload.get("system_ui")
    required = {"immersive_or_fullscreen", "navigation_bar_visible"}
    if not isinstance(value, dict) or set(value) != required:
        return
    if all(
        isinstance(value[field], bool) or value[field] == "unknown"
        for field in required
    ):
        return
    payload["system_ui"] = {
        "immersive_or_fullscreen": "unknown",
        "navigation_bar_visible": "unknown",
    }


def _normalize_tab_navigation_safety(
    payload: dict[str, Any],
    goal_context: dict[str, Any],
) -> None:
    """Remove destructive/aggregate candidates from an open-tab goal.

    This never creates a target or changes bounds. It only prevents a tab's
    close glyph and the surrounding tab-group container from competing with
    the visually reported tab card when the goal is safe navigation.
    """

    visible_goal = json.dumps(
        goal_context,
        ensure_ascii=False,
        separators=(",", ":"),
    ).casefold()
    tab_goal = any(
        marker in visible_goal
        for marker in ("标签页", "页签", "tab", "window card")
    )
    opens_tab = any(
        marker in visible_goal
        for marker in ("打开", "切换", "进入", "open", "switch", "enter")
    )
    closes_tab = any(
        marker in visible_goal
        for marker in ("关闭", "删除", "close", "remove", "delete")
    )
    if not tab_goal or not opens_tab or closes_tab:
        return
    elements = payload.get("elements")
    if not isinstance(elements, list):
        return
    for item in elements:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().casefold()
        meaning = str(item.get("meaning") or "").strip().casefold()
        if meaning in {"close_tab", "remove_tab", "delete_tab"} or (
            role == "container" and meaning in {"tab_group", "tabs", "tab_container"}
        ):
            states = item.get("states")
            if isinstance(states, dict):
                states["goal_relevant"] = False


def _active_subgoal_visual_context(context: dict[str, Any]) -> dict[str, Any]:
    """Return the current graph node's observation focus when it is present."""

    entities = context.get("entities")
    if not isinstance(entities, dict):
        return context
    focus = entities.get("active_subgoal_visual_context")
    if not isinstance(focus, dict):
        return context
    required = {
        "subgoal_id",
        "objective",
        "constraints",
        "completion_conditions",
        "execution_class",
        "goal_entities",
    }
    if set(focus) != required:
        return context
    if (
        not str(focus.get("subgoal_id") or "").strip()
        or not str(focus.get("objective") or "").strip()
        or not isinstance(focus.get("constraints"), list)
        or not isinstance(focus.get("completion_conditions"), list)
        or not isinstance(focus.get("goal_entities"), dict)
    ):
        return context
    return focus


def _observation_goal_context(context: dict[str, Any]) -> dict[str, Any]:
    """Expose only the active graph node to model evidence-selection prompts.

    The root objective can describe several future actions.  Supplying that full
    workflow while a different node is active makes an observation model select
    evidence for later steps.  The graph node is therefore the only semantic
    focus once the orchestrator has supplied its strict visual context.  App and
    controller authority still come from the local scene and policy layers.
    """

    focused = _active_subgoal_visual_context(context)
    if focused is context:
        return context
    return {
        "subgoal_id": focused["subgoal_id"],
        "objective": focused["objective"],
        "constraints": list(focused["constraints"]),
        "completion_conditions": list(focused["completion_conditions"]),
        "execution_class": focused["execution_class"],
        "goal_entities": dict(focused["goal_entities"]),
    }


def _goal_requests_page_title(context: dict[str, Any]) -> bool:
    text = json.dumps(
        _active_subgoal_visual_context(context),
        ensure_ascii=False,
        separators=(",", ":"),
    ).casefold()
    return any(marker in text for marker in ("标题", "题头", "heading", "title"))


def _normalize_page_title_identity(
    payload: dict[str, Any],
    goal_context: dict[str, Any],
) -> None:
    """Promote only one explicit, strong page-title fact into screen identity."""

    if not _goal_requests_page_title(goal_context):
        return
    if str(payload.get("screen_id") or "").strip().casefold() != "unknown":
        return
    elements = payload.get("elements")
    if not isinstance(elements, list):
        return
    candidates = []
    for item in elements:
        if not isinstance(item, dict):
            continue
        meaning = str(item.get("meaning") or "").strip().casefold().replace("_", " ")
        states = item.get("states")
        bounds = item.get("bounds")
        label = str(item.get("label") or "").strip()
        confidence = item.get("confidence")
        if (
            item.get("role") != "text"
            or meaning not in {"page title", "screen title", "page heading", "heading"}
            or not isinstance(states, dict)
            or states.get("goal_relevant") is not True
            or states.get("fully_visible") is not True
            or not isinstance(bounds, list)
            or len(bounds) != 4
            or not label
            or len(label) > 120
            or not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or float(confidence) < 0.90
        ):
            continue
        try:
            top = float(bounds[1]) / 1000.0
            height = (float(bounds[3]) - float(bounds[1])) / 1000.0
        except (TypeError, ValueError):
            continue
        if top < 0.0 or top > 0.45 or height <= 0.0 or height > 0.15:
            continue
        candidates.append(label)
    if len(candidates) == 1:
        payload["screen_id"] = re.sub(r"\s+", " ", candidates[0]).strip()


def _scene_has_grounded_page_title(scene: UIScene) -> bool:
    candidates = []
    for element in scene.elements:
        meaning = element.meaning.casefold().replace("_", " ")
        if (
            element.role == "text"
            and meaning in {"page title", "screen title", "page heading", "heading"}
            and element.label.strip()
            and float(element.confidence) >= 0.90
            and element.states.get("goal_relevant") is True
            and element.states.get("fully_visible") is True
            and 0.0 <= element.bounds[1] <= 0.45
            and 0.0 < element.bounds[3] - element.bounds[1] <= 0.15
        ):
            candidates.append(element)
    return bool(
        len(candidates) == 1
        and scene.screen_id.casefold() != "unknown"
        and re.sub(r"\s+", " ", candidates[0].label).strip() == scene.screen_id
    )


def _apply_system_ui_audit(
    scene: UIScene,
    raw: str,
    *,
    fingerprint: str,
) -> tuple[UIScene, float, tuple[str, ...]]:
    facts, confidence, evidence = _parse_system_ui_audit(raw)
    audited_scene = replace(
        scene,
        system_ui=facts,
        confidence=max(float(scene.confidence), confidence),
        fingerprint=fingerprint,
    )
    audited_scene.validate()
    return audited_scene, confidence, evidence


def _parse_system_ui_audit(
    raw: str,
) -> tuple[SystemUIFacts, float, tuple[str, ...]]:
    try:
        payload = _extract_json_object(raw)
        required = {
            "protocol_version",
            "immersive_or_fullscreen",
            "navigation_bar_visible",
            "confidence",
            "evidence",
        }
        if set(payload) != required:
            missing = sorted(required - set(payload))
            unexpected = sorted(set(payload) - required)
            details = []
            if missing:
                details.append("缺少字段：" + ", ".join(missing))
            if unexpected:
                details.append("包含协议外字段：" + ", ".join(map(str, unexpected)))
            raise UISceneError("系统界面审计结构无效；" + "；".join(details))
        if payload["protocol_version"] != SYSTEM_UI_AUDIT_VERSION:
            raise UISceneError("系统界面审计协议版本不匹配。")

        facts = SystemUIFacts(
            immersive_or_fullscreen=payload["immersive_or_fullscreen"],
            navigation_bar_visible=payload["navigation_bar_visible"],
        )
        facts.validate()
        if not isinstance(facts.immersive_or_fullscreen, bool) or not isinstance(
            facts.navigation_bar_visible, bool
        ):
            raise UISceneError("系统界面审计未同时给出两个明确布尔事实。")

        confidence_value = payload["confidence"]
        if isinstance(confidence_value, bool) or not isinstance(
            confidence_value, (int, float)
        ):
            raise UISceneError("系统界面审计置信度格式无效。")
        confidence = float(confidence_value)
        if not 0.0 <= confidence <= 1.0:
            raise UISceneError("系统界面审计置信度必须在0到1之间。")
        if confidence < MIN_SYSTEM_UI_AUDIT_CONFIDENCE:
            raise UISceneError("系统界面审计置信度不足，无法授权系统导航动作。")

        raw_evidence = payload["evidence"]
        if not isinstance(raw_evidence, list) or not 1 <= len(raw_evidence) <= 2:
            raise UISceneError("系统界面审计 evidence 必须包含一到两个短字符串。")
        evidence: list[str] = []
        forbidden = re.compile(
            r"(?:coordinates?|coords?|bounds?|\bx\s*[=:]|\by\s*[=:]|"
            r"\bpx\s*:|\bmm\s*:|\(\s*\d+\s*,\s*\d+\s*\)|"
            r"\b(?:tap|click|press|swipe|drag|execute|suggest)\b|"
            r"点击|滑动|拖动|按下|坐标|执行|建议)",
            re.IGNORECASE,
        )
        for item in raw_evidence:
            if not isinstance(item, str):
                raise UISceneError("系统界面审计 evidence 只允许字符串。")
            text = item.strip()
            if not text or len(text) > 160:
                raise UISceneError("系统界面审计 evidence 字符串为空或过长。")
            if forbidden.search(text):
                raise UISceneError("系统界面审计 evidence 包含坐标或控制指令。")
            evidence.append(text)
        return facts, confidence, tuple(evidence)
    except (UISceneError, ValueError, TypeError) as exc:
        raise VisionAgentError(f"系统界面只读审计不符合协议：{exc}") from exc


def _normalize_unique_input_focus(payload: dict[str, Any]) -> None:
    """Derive focus only from one target input plus visible soft keyboard facts."""

    elements = payload.get("elements")
    if not isinstance(elements, list):
        return
    candidates = [
        item
        for item in elements
        if isinstance(item, dict)
        and str(item.get("role") or "").strip() == "input"
        and isinstance(item.get("states"), dict)
        and item["states"].get("goal_relevant") is True
        and float(item.get("confidence") or 0.0) >= MIN_TARGET_CONFIDENCE
    ]
    if len(candidates) != 1:
        return
    visible_text = " ".join(
        [
            str(payload.get("summary") or ""),
            *(
                str(value)
                for value in (payload.get("overlays") or [])
                if isinstance(value, str)
            ),
            *(
                " ".join(
                    str(item.get(key) or "")
                    for key in ("role", "meaning", "label")
                )
                for item in elements
                if isinstance(item, dict)
            ),
        ]
    ).casefold()
    keyboard_visible = bool(
        re.search(
            r"(?:软键盘|输入法|键盘|keyboard|ime)",
            visible_text,
            re.IGNORECASE,
        )
    )
    if not keyboard_visible:
        return
    candidate = candidates[0]
    states = dict(candidate.get("states") or {})
    if states.get("focused") is False:
        # An explicit contradictory visual fact always wins.
        return
    states["focused"] = True
    candidate["states"] = states


def _normalize_local_text_clear_structure(
    payload: dict[str, Any],
    goal_context: dict[str, Any],
) -> None:
    """Bind one observed clear control to one non-empty input for clear goals.

    This normalization grants no action permission.  It only restores omitted
    goal-relevance facts when the model has already reported the complete
    high-confidence structure required by the controller.  Focus is still
    derived separately and only when the same scene also reports a visible
    soft keyboard.
    """

    if not _goal_requests_local_text_clear(goal_context):
        return
    elements = payload.get("elements")
    if not isinstance(elements, list):
        return

    all_inputs = [
        item
        for item in elements
        if isinstance(item, dict)
        and str(item.get("role") or "").strip() == "input"
        and isinstance(item.get("states"), dict)
        and isinstance(item["states"].get("value"), str)
        and bool(item["states"]["value"])
        and isinstance(item.get("confidence"), (int, float))
        and not isinstance(item.get("confidence"), bool)
        and float(item["confidence"]) >= 0.9
        and _valid_1000_bounds(item.get("bounds"))
    ]
    claimed_clear_controls = [
        item
        for item in elements
        if isinstance(item, dict)
        and str(item.get("role") or "").strip() in {"button", "icon"}
        and isinstance(item.get("states"), dict)
        and (
            str(item.get("meaning") or "").strip() == "clear_local_text"
            or item["states"].get("local_text_clear") is True
        )
    ]
    inputs = [
        item
        for item in all_inputs
        if item["states"].get("goal_relevant") is not False
    ]
    clear_controls = [
        item
        for item in claimed_clear_controls
        if item["states"].get("local_text_clear") is True
        and item["states"].get("goal_relevant") is not False
        and isinstance(item.get("confidence"), (int, float))
        and not isinstance(item.get("confidence"), bool)
        and float(item["confidence"]) >= 0.9
        and _valid_1000_bounds(item.get("bounds"))
        and not _has_cancel_semantics(item)
        and _has_exact_clear_glyph(item)
    ]
    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for input_element in inputs:
        il, it, ir, ib = (float(value) for value in input_element["bounds"])
        input_height = max(1e-9, ib - it)
        for clear_control in clear_controls:
            cl, ct, cr, cb = (float(value) for value in clear_control["bounds"])
            clear_width = cr - cl
            clear_height = max(1e-9, cb - ct)
            vertical_overlap = max(0.0, min(ib, cb) - max(it, ct))
            gap = max(0.0, cl - ir)
            geometrically_bound = (
                vertical_overlap / clear_height >= 0.6
                and cl >= il + 0.4 * (ir - il)
                and cr <= min(1000.0, ir + 200.0)
                and gap <= max(40.0, input_height)
                and clear_width <= 2.0 * input_height
                and clear_height <= 1.5 * input_height
            )
            if geometrically_bound:
                matches.append((input_element, clear_control))

    if len(matches) == 1 and len(inputs) == 1 and len(clear_controls) == 1:
        input_element, clear_control = matches[0]
        input_element["states"] = dict(input_element["states"])
        input_element["states"]["goal_relevant"] = True
        clear_control["states"] = dict(clear_control["states"])
        clear_control["states"]["goal_relevant"] = True
        return

    # Fail closed and let targeted refinement re-observe the exact visual
    # structure. A model semantic claim alone grants no clear-button authority.
    for item in all_inputs:
        item["states"] = dict(item["states"])
        item["states"]["goal_relevant"] = False
        item["states"].pop("focused", None)
    for item in claimed_clear_controls:
        item["states"] = dict(item["states"])
        item["states"].pop("local_text_clear", None)
        item["states"]["goal_relevant"] = False


def _has_cancel_semantics(item: dict[str, Any]) -> bool:
    visible = " ".join(
        [
            str(item.get("meaning") or ""),
            str(item.get("label") or ""),
            *(str(value) for value in (item.get("evidence") or [])),
        ]
    ).casefold()
    return "取消" in visible or bool(re.search(r"\bcancel(?:led|ing)?\b", visible))


def _has_exact_clear_glyph(item: dict[str, Any]) -> bool:
    return str(item.get("label") or "").strip().casefold() in {"×", "✕", "✖", "x"}


def _goal_directed_roi_bounds(
    context: dict[str, Any],
) -> tuple[int, int, int, int] | None:
    """Select a coarse ROI only from the typed, user-authored spatial hint."""

    hints: list[str] = []
    focused = _active_subgoal_visual_context(context)
    for source in (focused, context):
        if not isinstance(source, dict):
            continue
        for key in ("goal_entities", "entities"):
            entities = source.get(key)
            if not isinstance(entities, dict):
                continue
            hint = entities.get("spatial_hint")
            if isinstance(hint, str) and hint.strip():
                hints.append(hint.strip().casefold())
    if len(set(hints)) != 1:
        return None
    visible = hints[0]
    top = any(term in visible for term in ("顶部", "上方", "顶端", "top"))
    bottom = any(term in visible for term in ("底部", "下方", "底端", "bottom"))
    left = any(term in visible for term in ("左侧", "左边", "left"))
    right = any(term in visible for term in ("右侧", "右边", "right"))
    if top and bottom:
        top = bottom = False
    if left and right:
        left = right = False
    horizontal = (0, 1000)
    vertical = (0, 1000)
    if left:
        horizontal = (0, 560)
    elif right:
        horizontal = (440, 1000)
    if top:
        vertical = (0, 420)
    elif bottom:
        vertical = (580, 1000)
    if horizontal == (0, 1000) and vertical == (0, 1000):
        return None
    return horizontal[0], vertical[0], horizontal[1], vertical[1]


def _unique_payload_input_bounds(
    payload: dict[str, Any],
) -> tuple[int, int, int, int] | None:
    """Keep one compact input box only as non-authoritative crop evidence."""

    elements = payload.get("elements")
    if not isinstance(elements, list):
        return None
    inputs = [
        item
        for item in elements
        if isinstance(item, dict)
        and str(item.get("role") or "").strip() == "input"
        and _valid_1000_bounds(item.get("bounds"))
    ]
    if len(inputs) != 1:
        return None
    return tuple(round(float(part)) for part in inputs[0]["bounds"])


def _input_audit_retry_roi(
    context: dict[str, Any],
    *,
    preliminary_input_bounds_hint: tuple[int, int, int, int] | None,
    first_audit_raw: str | None = None,
) -> tuple[int, int, int, int] | None:
    """Choose one detail crop without granting compact geometry authority."""

    goal_roi = _goal_directed_roi_bounds(context)
    if goal_roi is not None:
        return goal_roi
    if (
        not _goal_requests_input(context)
        or preliminary_input_bounds_hint is None
    ):
        return None
    field_height = max(
        1,
        preliminary_input_bounds_hint[3] - preliminary_input_bounds_hint[1],
    )
    # A coarse box can lag behind a keyboard-induced pan and can cover only
    # the editable interior of a tall multiline field. Keep proportional
    # context above it so the independent retry can prove all four field edges.
    field_upward_context = max(240, min(360, field_height * 2))
    top_candidates = [
        preliminary_input_bounds_hint[1] - field_upward_context
    ]
    # Focusing an input can pan the App content upward while a compact scene
    # still reports the pre-focus field box.  When the first dedicated audit
    # independently proves a complete keyboard, use only that broad keyboard
    # geometry to widen the retry crop above it.  The keyboard hint never
    # mints an input target or executable coordinates; the retry must still
    # re-establish the complete field and keyboard in its own local audit.
    try:
        first_payload = (
            _extract_json_object(first_audit_raw)
            if isinstance(first_audit_raw, str) and first_audit_raw.strip()
            else None
        )
    except (VisionAgentError, ValueError, TypeError):
        first_payload = None
    if (
        isinstance(first_payload, dict)
        and first_payload.get("protocol_version")
        == INPUT_STRUCTURE_AUDIT_VERSION
    ):
        keyboard = first_payload.get("keyboard")
        keyboard_bounds = (
            keyboard.get("bounds") if isinstance(keyboard, dict) else None
        )
        if (
            isinstance(keyboard, dict)
            and keyboard.get("visible") is True
            and _valid_1000_bounds(keyboard_bounds)
        ):
            left, keyboard_top, right, keyboard_bottom = (
                round(float(value)) for value in keyboard_bounds
            )
            keyboard_width = right - left
            keyboard_height = keyboard_bottom - keyboard_top
            if keyboard_width >= 300 and keyboard_height >= 180:
                upward_context = max(
                    240,
                    min(420, round(keyboard_height * 0.8)),
                )
                top_candidates.append(keyboard_top - upward_context)
    top = max(0, min(top_candidates))
    # A crop must materially improve resolution and must include the complete
    # keyboard below the coarse field.  The dedicated audit independently
    # re-establishes all actionable bounds inside the crop.
    if top < 100:
        return None
    return (0, top, 1000, 1000)


def _crop_normalized(
    image: Image.Image,
    bounds: tuple[int, int, int, int],
) -> Image.Image:
    left, top, right, bottom = bounds
    x0 = round(left * image.width / 1000)
    y0 = round(top * image.height / 1000)
    x1 = round(right * image.width / 1000)
    y1 = round(bottom * image.height / 1000)
    return image.crop((x0, y0, x1, y1))


def _roi_observation_note(
    bounds: tuple[int, int, int, int] | None,
) -> str:
    if bounds is None:
        return "本次仍提供完整手机画面；bounds相对于完整画面。"
    return (
        f"本次依次提供完整手机画面和根据目标明确方位词裁出的高清局部，局部在原图范围为{list(bounds)}。"
        "第一张只用于理解页面上下文；必须在第二张高清局部中重新辨认目标。"
        "第二张只提供放大细节，绝不能作为坐标系；所有bounds必须回到第一张完整手机画面，"
        "相对于第一张使用0..1000坐标。"
        "局部图只用于看清事实，不增加任何动作权限。"
    )


def _normalize_prefilled_input_structure(
    payload: dict[str, Any],
    goal_context: dict[str, Any],
) -> None:
    """Infer an input only from a strict model-reported field/button structure."""

    if not _goal_requests_input(goal_context):
        return
    elements = payload.get("elements")
    if not isinstance(elements, list) or any(
        isinstance(item, dict) and str(item.get("role") or "").strip() == "input"
        for item in elements
    ):
        return

    def trusted(item: Any, role: str) -> bool:
        return (
            isinstance(item, dict)
            and str(item.get("role") or "").strip() == role
            and isinstance(item.get("confidence"), (int, float))
            and not isinstance(item.get("confidence"), bool)
            and float(item["confidence"]) >= 0.9
            and _valid_1000_bounds(item.get("bounds"))
            and isinstance(item.get("states"), dict)
            and item["states"].get("fully_visible") is True
        )

    containers = [
        item
        for item in elements
        if trusted(item, "container")
        and isinstance(item.get("states"), dict)
        and item["states"].get("goal_relevant") is True
        and _has_any_semantic_term(item, ("search", "query", "input", "form", "搜索", "查询", "输入"))
    ]
    texts = [
        item
        for item in elements
        if trusted(item, "text")
        and isinstance(item.get("states"), dict)
        and item["states"].get("goal_relevant") is True
    ]
    buttons = [
        item
        for item in elements
        if trusted(item, "button")
        and isinstance(item.get("states"), dict)
        and item["states"].get("goal_relevant") is False
        and _has_any_semantic_term(item, ("search", "submit", "go", "搜索", "提交", "查找"))
    ]
    matches: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for container in containers:
        cb = tuple(float(value) for value in container["bounds"])
        if cb[2] - cb[0] < 240 or cb[3] - cb[1] > 300:
            continue
        for button in buttons:
            bb = tuple(float(value) for value in button["bounds"])
            button_inside_group = (
                _bounds_inside(bb, cb, tolerance=35)
                and bb[0] > cb[0] + 0.45 * (cb[2] - cb[0])
            )
            button_adjacent_right = (
                bb[0] >= cb[2] - 0.15 * (cb[2] - cb[0])
                and bb[0] <= cb[2] + 80
                and bb[2] > cb[2]
            )
            if not (button_inside_group or button_adjacent_right):
                continue
            if _vertical_overlap_ratio(bb, cb) < 0.65:
                continue
            for text in texts:
                tb = tuple(float(value) for value in text["bounds"])
                if not _bounds_inside(tb, cb, tolerance=35) or tb[2] > bb[0] + 20:
                    continue
                if _vertical_overlap_ratio(tb, cb) < 0.45:
                    continue
                matches.append((container, text, button))
    if len(matches) != 1:
        return
    container, text, button = matches[0]
    cb = tuple(float(value) for value in container["bounds"])
    bb = tuple(float(value) for value in button["bounds"])
    input_bounds = [round(cb[0]), round(cb[1]), round(min(cb[2], bb[0])), round(cb[3])]
    if input_bounds[2] - input_bounds[0] < 120:
        return
    normalized_input_box = tuple(float(value) for value in input_bounds)
    for item in elements:
        if not isinstance(item, dict) or item is button:
            continue
        raw_item_bounds = item.get("bounds")
        if item in (container, text) or (
            _valid_1000_bounds(raw_item_bounds)
            and _bounds_inside(
                tuple(float(value) for value in raw_item_bounds),
                normalized_input_box,
                tolerance=20,
            )
        ):
            item["states"] = dict(item.get("states") or {})
            item["states"]["goal_relevant"] = False
    label = str(text.get("label") or "").strip()[:200]
    evidence = []
    for source in (container, text, button):
        for value in source.get("evidence") or []:
            value = str(value).strip()
            if value and value not in evidence:
                evidence.append(value[:200])
    elements.append(
        {
            "element_id": "local_structured_input_1",
            "role": "input",
            "meaning": "prefilled_text_input",
            "label": label,
            "bounds": input_bounds,
            "confidence": min(
                float(container["confidence"]),
                float(text["confidence"]),
                float(button["confidence"]),
            ),
            "states": {"goal_relevant": True, "value": label},
            "evidence": evidence[:6],
        }
    )


def _valid_1000_bounds(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    if not all(
        isinstance(part, (int, float)) and not isinstance(part, bool)
        for part in value
    ):
        return False
    left, top, right, bottom = (float(part) for part in value)
    return 0 <= left < right <= 1000 and 0 <= top < bottom <= 1000


def _typed_prefix_input_survives_invalid_keyboard_geometry(
    application_inputs: Any,
    goal_context: dict[str, Any],
) -> bool:
    """Keep one typed prefix as verification-only evidence.

    A post-action audit can read the exact application value while returning
    unusable keyboard geometry. The field value is independent evidence, but
    it may survive only as a non-actionable typed prefix. No keyboard fact or
    coordinate from the malformed portion is retained.
    """

    target_text = _goal_active_input_transaction_text(goal_context)
    field_id, field_label, _multiline = _goal_active_input_field(goal_context)
    if (
        not target_text
        or not field_id
        or not isinstance(application_inputs, list)
        or len(application_inputs) != 1
    ):
        return False
    item = application_inputs[0]
    required = {
        "structure_id",
        "bounds",
        "fully_visible",
        "text",
        "placeholder",
        "visible_editable_cues",
        "caret_line_index",
        "confidence",
        "right_button",
    }
    if (
        not isinstance(item, dict)
        or not required.issubset(item)
        or set(item) - required - {"field_labels"}
        or item.get("fully_visible") is not True
        or not _valid_1000_bounds(item.get("bounds"))
    ):
        return False
    confidence = item.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or float(confidence) < 0.9
    ):
        return False
    observed = item.get("text")
    if (
        not isinstance(observed, str)
        or not observed
        or observed == target_text
        or not target_text.startswith(observed)
    ):
        return False
    cues = item.get("visible_editable_cues")
    if (
        not isinstance(cues, list)
        or not cues
        or len(cues) > 4
        or any(not isinstance(value, str) or not value.strip() for value in cues)
    ):
        return False
    labels = item.get("field_labels", [])
    if (
        not isinstance(labels, list)
        or len(labels) > 6
        or any(not isinstance(value, str) or not value.strip() for value in labels)
    ):
        return False
    if field_label and sum(
        value.strip().casefold() == field_label.casefold() for value in labels
    ) != 1:
        return False
    return True


def _strip_preliminary_elements_for_keyboard_mode_audit(
    payload: dict[str, Any],
    goal_context: dict[str, Any],
) -> None:
    """Make the strict input audit the sole element authority for mode goals.

    An explicit keyboard-mode goal always runs the independent full-frame input
    structure audit. Compact elements therefore cannot authorize or block that
    audit based on any model-authored role, meaning, label, state, or coordinate
    frame. If every element uses the standard passive scene shape and contains
    no action-like field, the whole preliminary collection is discarded without
    reading or reusing any value. Otherwise it remains for strict validation to
    fail closed. Scene identity, system UI and camera alignment remain intact.
    """

    if not _goal_requests_keyboard_mode_switch(goal_context):
        return
    elements = payload.get("elements")
    if not isinstance(elements, list):
        return
    exact_fields = {
        "element_id",
        "role",
        "meaning",
        "label",
        "bounds",
        "confidence",
        "states",
        "evidence",
    }
    action_like = {
        "action",
        "actions",
        "plan",
        "step",
        "steps",
        "tap",
        "swipe",
        "command",
        "coordinates",
    }

    def contains_action_like_key(value: Any) -> bool:
        if isinstance(value, dict):
            return any(
                str(key).strip().casefold() in action_like
                or contains_action_like_key(part)
                for key, part in value.items()
            )
        if isinstance(value, list):
            return any(contains_action_like_key(part) for part in value)
        return False

    for item in elements:
        if (
            not isinstance(item, dict)
            or set(item) != exact_fields
            or contains_action_like_key(item)
        ):
            return
    payload["elements"] = []


def _strip_preliminary_input_geometry_for_dedicated_audit(
    payload: dict[str, Any],
    goal_context: dict[str, Any],
) -> bool:
    """Remove passive compact input proposals before the strict input audit.

    For an active input subgoal, the later full-frame input-structure audit is
    the sole geometry authority. Compact input boxes are therefore evidence
    selection hints only and cannot block that audit merely because the vision
    model used source-pixel or width-scaled coordinates. Only exact passive
    scene elements are removable; an input carrying an action-like field,
    malformed states/evidence, or a nonstandard shape remains for strict parsing
    to reject. Non-input page identity and navigation facts are preserved.
    """

    if not _goal_requests_input(goal_context):
        return False
    elements = payload.get("elements")
    if not isinstance(elements, list):
        return False
    exact_fields = {
        "element_id",
        "role",
        "meaning",
        "label",
        "bounds",
        "confidence",
        "states",
        "evidence",
    }
    action_like = {
        "action",
        "actions",
        "plan",
        "step",
        "steps",
        "tap",
        "swipe",
        "command",
        "coordinates",
    }

    def contains_action_like_key(value: Any) -> bool:
        if isinstance(value, dict):
            return any(
                str(key).strip().casefold() in action_like
                or contains_action_like_key(part)
                for key, part in value.items()
            )
        if isinstance(value, list):
            return any(contains_action_like_key(part) for part in value)
        return False

    retained: list[Any] = []
    isolated = False
    for item in elements:
        removable_input = (
            isinstance(item, dict)
            and set(item) == exact_fields
            and str(item.get("role") or "").strip() == "input"
            and isinstance(item.get("bounds"), list)
            and len(item["bounds"]) == 4
            and isinstance(item.get("states"), dict)
            and isinstance(item.get("evidence"), list)
            and not contains_action_like_key(item)
        )
        if removable_input:
            isolated = True
            continue
        retained.append(item)
    if isolated:
        payload["elements"] = retained
    return isolated


def _strip_preliminary_keyboard_containers_for_dedicated_audit(
    payload: dict[str, Any],
    goal_context: dict[str, Any],
) -> bool:
    """Remove compact keyboard geometry before the sole strict input audit.

    During an explicit text-input goal the later independent input-structure
    audit is the sole authority for keyboard layout, mode and key geometry.
    Compact observation may still redundantly emit a keyboard container or a
    keyboard-specific control.  Keeping either copy lets malformed compact
    geometry veto the dedicated audit before it can run.  Discard only exact
    scene elements whose typed meaning unambiguously belongs to that dedicated
    keyboard catalog.  Action-bearing or otherwise ambiguous elements remain
    so the normal scene parser still fails closed.
    """

    if not _goal_has_explicit_input_text(goal_context):
        return False
    elements = payload.get("elements")
    if not isinstance(elements, list):
        return False
    exact_fields = {
        "element_id",
        "role",
        "meaning",
        "label",
        "bounds",
        "confidence",
        "states",
        "evidence",
    }
    action_like = {
        "action",
        "actions",
        "plan",
        "step",
        "steps",
        "tap",
        "swipe",
        "command",
        "coordinates",
    }

    def contains_action_like_key(value: Any) -> bool:
        if isinstance(value, dict):
            return any(
                str(key).strip().casefold() in action_like
                or contains_action_like_key(part)
                for key, part in value.items()
            )
        if isinstance(value, list):
            return any(contains_action_like_key(part) for part in value)
        return False

    retained: list[Any] = []
    isolated = False
    dedicated_keyboard_meanings = {
        "input_exact_enter_key",
        "input_exact_literal_key",
        "switch_keyboard_case",
        "switch_keyboard_input_mode",
        "switch_keyboard_layout",
    }
    for item in elements:
        states = item.get("states") if isinstance(item, dict) else None
        meaning = (
            str(item.get("meaning") or "").strip().casefold()
            if isinstance(item, dict)
            else ""
        )
        meaning_tokens = {
            token
            for token in re.split(r"[^a-z0-9]+", meaning)
            if token
        }
        confidence = item.get("confidence") if isinstance(item, dict) else None
        removable_keyboard_container = (
            isinstance(item, dict)
            and set(item) == exact_fields
            and isinstance(item.get("element_id"), str)
            and bool(item["element_id"].strip())
            and str(item.get("role") or "").strip() == "container"
            and isinstance(item.get("meaning"), str)
            and "keyboard" in meaning_tokens
            and isinstance(item.get("label"), str)
            and _valid_1000_bounds(item.get("bounds"))
            and not isinstance(confidence, bool)
            and isinstance(confidence, (int, float))
            and math.isfinite(float(confidence))
            and 0.0 <= float(confidence) <= 1.0
            and isinstance(states, dict)
            and set(states).issubset(
                {
                    "goal_relevant",
                    "fully_visible",
                    "keyboard_layout",
                    "keyboard_input_mode",
                }
            )
            and (
                "keyboard_layout" in states
                or "keyboard_input_mode" in states
            )
            and (
                "goal_relevant" not in states
                or isinstance(states["goal_relevant"], bool)
            )
            and (
                "fully_visible" not in states
                or isinstance(states["fully_visible"], bool)
            )
            and (
                "keyboard_layout" not in states
                or states["keyboard_layout"]
                in {"qwerty", "numeric", "symbol", "unknown"}
            )
            and (
                "keyboard_input_mode" not in states
                or states["keyboard_input_mode"]
                in {"direct_latin", "chinese_pinyin", "unknown"}
            )
            and isinstance(item.get("evidence"), list)
            and all(isinstance(part, str) for part in item["evidence"])
            and not contains_action_like_key(item)
        )
        removable_keyboard_control = (
            isinstance(item, dict)
            and set(item) == exact_fields
            and isinstance(item.get("element_id"), str)
            and bool(item["element_id"].strip())
            and str(item.get("role") or "").strip() in {"button", "key"}
            and meaning in dedicated_keyboard_meanings
            and isinstance(item.get("label"), str)
            and isinstance(states, dict)
            and isinstance(item.get("evidence"), list)
            and all(isinstance(part, str) for part in item["evidence"])
            and not contains_action_like_key(item)
        )
        if removable_keyboard_container or removable_keyboard_control:
            isolated = True
            continue
        retained.append(item)
    if isolated:
        payload["elements"] = retained
    return isolated


def _drop_out_of_range_non_goal_elements(
    payload: dict[str, Any],
    goal_context: dict[str, Any] | None = None,
) -> None:
    """Discard only explicitly non-goal peripheral elements with invalid bounds.

    Model-authored goal candidates and elements without an explicit
    ``goal_relevant: false`` assertion remain strict and still fail closed. An
    invalid input remains strict for an active input subgoal, but may be removed
    after a unique literal target has locally made it an explicit non-goal
    peripheral for the current non-input subgoal. Dropping such a peripheral can
    only remove information; it never creates a target or converts pixel
    coordinates into actionable coordinates.
    """

    elements = payload.get("elements")
    if not isinstance(elements, list):
        return
    retained: list[Any] = []
    for item in elements:
        if not isinstance(item, dict) or _valid_1000_bounds(item.get("bounds")):
            retained.append(item)
            continue
        states = item.get("states")
        role = str(item.get("role") or "").strip()
        safe_to_discard = (
            isinstance(states, dict)
            and states.get("goal_relevant") is False
            and (
                role != "input"
                or not _goal_requests_input(goal_context or {})
            )
        )
        if not safe_to_discard:
            retained.append(item)
    payload["elements"] = retained


def _discard_compact_elements_for_targeted_geometry_recovery(
    payload: dict[str, Any],
    goal_context: dict[str, Any],
) -> bool:
    """Discard a whole passive compact element batch after target overflow.

    No coordinate is converted, clipped, or reused. The caller must force a
    fresh targeted delta whose complete geometry is validated independently.
    Input and keyboard-mode goals retain their stricter dedicated contracts.
    """

    if _goal_requests_input(goal_context) or _goal_requests_keyboard_mode_switch(
        goal_context
    ):
        return False
    elements = payload.get("elements")
    if not isinstance(elements, list) or not elements:
        return False
    exact_fields = {
        "element_id",
        "role",
        "meaning",
        "label",
        "bounds",
        "confidence",
        "states",
        "evidence",
    }
    action_like = {
        "action",
        "actions",
        "plan",
        "step",
        "steps",
        "tap",
        "swipe",
        "command",
        "coordinates",
    }

    def contains_action_like_key(value: Any) -> bool:
        if isinstance(value, dict):
            return any(
                str(key).strip().casefold() in action_like
                or contains_action_like_key(part)
                for key, part in value.items()
            )
        if isinstance(value, list):
            return any(contains_action_like_key(part) for part in value)
        return False

    target_overflow = False
    for item in elements:
        if (
            not isinstance(item, dict)
            or set(item) != exact_fields
            or contains_action_like_key(item)
            or not isinstance(item.get("states"), dict)
            or not isinstance(item.get("evidence"), list)
        ):
            return False
        bounds = item.get("bounds")
        if (
            not isinstance(bounds, list)
            or len(bounds) != 4
            or not all(
                isinstance(part, (int, float)) and not isinstance(part, bool)
                for part in bounds
            )
        ):
            return False
        left, top, right, bottom = (float(part) for part in bounds)
        if left < 0 or top < 0 or left >= right or top >= bottom:
            return False
        if (
            (right > 1000 or bottom > 1000)
            and item["states"].get("goal_relevant") is True
        ):
            target_overflow = True
    if not target_overflow:
        return False
    payload["elements"] = []
    return True


_ICON_CLUSTER_CLASSES = frozenset({"reload", "bookmark", "expand", "other"})
_ICON_CLUSTER_SHAPE_CUES = frozenset(
    {
        "curved_arc",
        "arrowhead",
        "circular_outline",
        "star",
        "ribbon_outline",
        "bookmark_outline",
        "four_corner_brackets",
        "expand_arrows",
        "other",
    }
)
_ICON_CLUSTER_RELOAD_FORBIDDEN_CUES = frozenset(
    {
        "star",
        "ribbon_outline",
        "bookmark_outline",
        "four_corner_brackets",
        "expand_arrows",
    }
)


def _strict_icon_cluster_audit_payload(raw: str) -> dict[str, Any]:
    payload = _extract_compact_json_object(raw)
    if set(payload) != {
        "protocol_version",
        "cluster_complete",
        "cluster_bounds",
        "controls",
    }:
        raise VisionAgentError("图标簇审计顶层字段不符合严格协议。")
    if payload.get("protocol_version") != ICON_CLUSTER_AUDIT_VERSION:
        raise VisionAgentError("图标簇审计协议版本无效。")
    if not isinstance(payload.get("cluster_complete"), bool):
        raise VisionAgentError("图标簇审计 cluster_complete 必须是布尔值。")
    controls = payload.get("controls")
    if not isinstance(controls, list) or len(controls) > 8:
        raise VisionAgentError("图标簇审计 controls 必须是至多8项的数组。")
    cluster_bounds = payload.get("cluster_bounds")
    if not controls:
        if payload["cluster_complete"] is not False or cluster_bounds is not None:
            raise VisionAgentError("空图标簇必须是不完整且 cluster_bounds=null。")
        return payload
    if not _valid_1000_bounds(cluster_bounds):
        raise VisionAgentError("图标簇审计 cluster_bounds 无效。")

    exact_control_fields = {
        "control_id",
        "semantic_class",
        "bounds",
        "confidence",
        "fully_visible",
        "single_glyph",
        "shape_cues",
    }
    seen_ids: set[str] = set()
    for control in controls:
        if not isinstance(control, dict) or set(control) != exact_control_fields:
            raise VisionAgentError("图标簇审计 control 字段不符合严格协议。")
        control_id = str(control.get("control_id") or "").strip()
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}", control_id)
            or control_id in seen_ids
        ):
            raise VisionAgentError("图标簇审计 control_id 为空、重复或格式无效。")
        seen_ids.add(control_id)
        if control.get("semantic_class") not in _ICON_CLUSTER_CLASSES:
            raise VisionAgentError("图标簇审计 semantic_class 不在允许列表。")
        if not _valid_1000_bounds(control.get("bounds")):
            raise VisionAgentError("图标簇审计 control bounds 无效。")
        confidence = control.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise VisionAgentError("图标簇审计 confidence 无效。")
        if not isinstance(control.get("fully_visible"), bool) or not isinstance(
            control.get("single_glyph"), bool
        ):
            raise VisionAgentError(
                "图标簇审计 fully_visible/single_glyph 必须是布尔值。"
            )
        cues = control.get("shape_cues")
        if (
            not isinstance(cues, list)
            or not cues
            or any(not isinstance(cue, str) for cue in cues)
            or len(cues) != len(set(cues))
            or not set(cues).issubset(_ICON_CLUSTER_SHAPE_CUES)
        ):
            raise VisionAgentError("图标簇审计 shape_cues 无效、重复或越出允许列表。")
    cluster_tuple = tuple(float(part) for part in cluster_bounds)
    if any(
        not _bounds_inside(
            tuple(float(part) for part in control["bounds"]),
            cluster_tuple,
            tolerance=0.0,
        )
        for control in controls
    ):
        # cluster_bounds is a redundant envelope, never an action target.  A
        # unique local repair is therefore possible without moving any model
        # control: rebuild only the envelope as the exact union of all already
        # validated control bounds. Compact-size checks still run afterward.
        payload = dict(payload)
        payload["cluster_bounds"] = [
            min(float(control["bounds"][0]) for control in controls),
            min(float(control["bounds"][1]) for control in controls),
            max(float(control["bounds"][2]) for control in controls),
            max(float(control["bounds"][3]) for control in controls),
        ]
    return payload


def _bounds_intersection_area(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return width * height


def _attestable_icon_cluster_bounds(
    payload: dict[str, Any],
    *,
    roi_bounds: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    """Return a compact full-frame cluster ROI eligible for local re-check."""

    controls = payload["controls"]
    reload_controls = [
        control for control in controls if control["semantic_class"] == "reload"
    ]
    candidate = reload_controls[0] if len(reload_controls) == 1 else None
    if payload["cluster_complete"] is not True or candidate is None:
        return None
    cluster_bounds = tuple(float(part) for part in payload["cluster_bounds"])
    width = cluster_bounds[2] - cluster_bounds[0]
    height = cluster_bounds[3] - cluster_bounds[1]
    if not (40.0 <= width <= 400.0 and 20.0 <= height <= 250.0):
        return None
    if roi_bounds is not None and not _bounds_inside(
        cluster_bounds,
        tuple(float(part) for part in roi_bounds),
        tolerance=20.0,
    ):
        return None
    candidate_bounds = tuple(float(part) for part in candidate["bounds"])
    if not (
        float(candidate["confidence"]) >= 0.90
        and candidate["fully_visible"] is True
        and candidate["single_glyph"] is True
        and {"curved_arc", "arrowhead"}.issubset(candidate["shape_cues"])
        and not _ICON_CLUSTER_RELOAD_FORBIDDEN_CUES.intersection(
            candidate["shape_cues"]
        )
    ):
        return None
    if any(
        _bounds_intersection_area(
            candidate_bounds,
            tuple(float(part) for part in other["bounds"]),
        )
        > 0.0
        for other in controls
        if other is not candidate
    ):
        return None
    outer_bounds = (
        tuple(float(part) for part in roi_bounds)
        if roi_bounds is not None
        else (0.0, 0.0, 1000.0, 1000.0)
    )
    horizontal_padding = max(20.0, width * 0.25)
    vertical_padding = max(20.0, height * 0.25)
    expanded = (
        max(outer_bounds[0], cluster_bounds[0] - horizontal_padding),
        max(outer_bounds[1], cluster_bounds[1] - vertical_padding),
        min(outer_bounds[2], cluster_bounds[2] + horizontal_padding),
        min(outer_bounds[3], cluster_bounds[3] + vertical_padding),
    )
    if expanded[2] - expanded[0] < width or expanded[3] - expanded[1] < height:
        return None
    return tuple(round(part) for part in expanded)


def _map_icon_cluster_audit_to_full_frame(
    payload: dict[str, Any],
    roi_bounds: tuple[int, int, int, int],
) -> dict[str, Any]:
    """Map the second audit's crop-local 0..1000 geometry to the full frame."""

    value = dict(payload)
    if payload["cluster_bounds"] is not None:
        value["cluster_bounds"] = list(
            map_roi_bounds_to_full(
                tuple(round(float(part)) for part in payload["cluster_bounds"]),
                roi_bounds,
            )
        )
    mapped_controls: list[dict[str, Any]] = []
    for control in payload["controls"]:
        mapped = dict(control)
        mapped["bounds"] = list(
            map_roi_bounds_to_full(
                tuple(round(float(part)) for part in control["bounds"]),
                roi_bounds,
            )
        )
        mapped_controls.append(mapped)
    value["controls"] = mapped_controls
    return value


def _snap_audited_input_to_local_border(
    frame: Image.Image,
    *,
    transform: Any,
    rough_bounds: tuple[float, float, float, float],
    audited_bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    """Tighten one model-attested input to one complete local border.

    The semantic authority still comes from the strict single-crop model audit.
    This local pass only replaces its loose geometry when exactly one broad,
    isolated border component is compatible with the rough or audited box. It
    uses no App identity, label, command text, screen coordinates, or fixture.
    Ambiguous and borderless UIs keep the model bounds and therefore retain the
    normal confirmation-time fail-closed comparison.
    """

    crop = transform.crop(frame.convert("RGB"))
    if crop.width < 80 or crop.height < 80:
        return None
    gray = crop.convert("L")
    # A focused field may sit above a light keyboard while the App itself is
    # dark (or vice versa), so a crop-wide background estimate is unstable.
    # Local max/min contrast detects the complete border in either polarity
    # without learning a theme, App, page, or coordinate.
    local_max = gray.filter(ImageFilter.MaxFilter(5))
    local_min = gray.filter(ImageFilter.MinFilter(5))
    edge_strength = ImageChops.subtract(local_max, local_min)
    mask = edge_strength.point(
        lambda value: 255 if value >= 25 else 0
    ).filter(ImageFilter.MaxFilter(3))
    pixels = mask.load()
    seen: set[tuple[int, int]] = set()
    components: list[tuple[int, int, int, int, int]] = []
    minimum_width = max(80, round(crop.width * 0.25))
    maximum_height = max(40, round(crop.height * 0.35))
    for y in range(crop.height):
        for x in range(crop.width):
            if not pixels[x, y] or (x, y) in seen:
                continue
            stack = [(x, y)]
            seen.add((x, y))
            xs: list[int] = []
            ys: list[int] = []
            while stack:
                current_x, current_y = stack.pop()
                xs.append(current_x)
                ys.append(current_y)
                for delta_y in (-1, 0, 1):
                    for delta_x in (-1, 0, 1):
                        neighbor_x = current_x + delta_x
                        neighbor_y = current_y + delta_y
                        neighbor = (neighbor_x, neighbor_y)
                        if (
                            0 <= neighbor_x < crop.width
                            and 0 <= neighbor_y < crop.height
                            and pixels[neighbor_x, neighbor_y]
                            and neighbor not in seen
                        ):
                            seen.add(neighbor)
                            stack.append(neighbor)
            left = min(xs)
            top = min(ys)
            right = max(xs) + 1
            bottom = max(ys) + 1
            component_width = right - left
            component_height = bottom - top
            area = component_width * component_height
            touches_edge = (
                left <= 1
                or top <= 1
                or right >= crop.width - 1
                or bottom >= crop.height - 1
            )
            density = len(xs) / max(1, area)
            if (
                not touches_edge
                and component_width >= minimum_width
                and 20 <= component_height <= maximum_height
                and component_width / max(1, component_height) >= 2.0
                and 0.005 <= density <= 0.45
            ):
                components.append((left, top, right, bottom, len(xs)))
    if not components:
        return None

    full_width, full_height = frame.size
    crop_left, crop_top, _, _ = transform.pixel_bounds

    def normalized(component: tuple[int, int, int, int, int]) -> tuple[float, float, float, float]:
        return (
            (crop_left + component[0]) / full_width,
            (crop_top + component[1]) / full_height,
            (crop_left + component[2]) / full_width,
            (crop_top + component[3]) / full_height,
        )

    def coverage(
        left_box: tuple[float, float, float, float],
        right_box: tuple[float, float, float, float],
    ) -> float:
        intersection = _bounds_intersection_area(left_box, right_box)
        left_area = max(0.0, left_box[2] - left_box[0]) * max(
            0.0, left_box[3] - left_box[1]
        )
        right_area = max(0.0, right_box[2] - right_box[0]) * max(
            0.0, right_box[3] - right_box[1]
        )
        return intersection / max(1e-9, min(left_area, right_area))

    ranked: list[tuple[float, float, float, tuple[float, float, float, float]]] = []
    for component in components:
        bounds = normalized(component)
        audited_coverage = coverage(bounds, audited_bounds)
        rough_coverage = coverage(bounds, rough_bounds)
        audited_center_delta = abs(
            (bounds[1] + bounds[3] - audited_bounds[1] - audited_bounds[3]) / 2.0
        )
        score = 3.0 * audited_coverage + 2.0 * rough_coverage - audited_center_delta
        ranked.append((score, audited_coverage, rough_coverage, bounds))
    ranked.sort(key=lambda item: item[0], reverse=True)
    best = ranked[0]
    directly_bound = max(best[1], best[2]) >= 0.15
    unique_broad_border = False
    if len(ranked) == 1:
        horizontal_overlap = max(
            0.0,
            min(best[3][2], rough_bounds[2]) - max(best[3][0], rough_bounds[0]),
        )
        smaller_width = min(
            best[3][2] - best[3][0], rough_bounds[2] - rough_bounds[0]
        )
        vertical_center_delta = abs(
            (best[3][1] + best[3][3] - rough_bounds[1] - rough_bounds[3]) / 2.0
        )
        unique_broad_border = bool(
            smaller_width > 0
            and horizontal_overlap / smaller_width >= 0.65
            and vertical_center_delta <= 0.30
        )
    separated = len(ranked) == 1 or best[0] - ranked[1][0] >= 0.25
    if not separated or not (directly_bound or unique_broad_border):
        return None
    return best[3]


def _snap_reload_audit_to_local_glyph(
    frame: Image.Image,
    payload: dict[str, Any],
    *,
    search_bounds: tuple[int, int, int, int],
) -> tuple[dict[str, Any] | None, tuple[int, int, int, int] | None]:
    """Bind model semantics to one isolated high-contrast glyph component.

    The model may identify the right icon while returning a loose or shifted
    box.  This check never invents semantics: it only replaces the sole audited
    reload box when that box overlaps exactly one isolated pixel component in
    the already-audited compact cluster.  Ambiguity remains fail-closed.
    """

    reload_controls = [
        control
        for control in payload["controls"]
        if control["semantic_class"] == "reload"
    ]
    if payload["cluster_complete"] is not True or len(reload_controls) != 1:
        return None, None
    candidate = reload_controls[0]
    candidate_bounds = tuple(float(part) for part in candidate["bounds"])

    width, height = frame.size
    left, top, right, bottom = search_bounds
    pixel_search = (
        max(0, min(width - 1, round(left * width / 1000))),
        max(0, min(height - 1, round(top * height / 1000))),
        max(1, min(width, round(right * width / 1000))),
        max(1, min(height, round(bottom * height / 1000))),
    )
    if pixel_search[2] - pixel_search[0] < 12 or pixel_search[3] - pixel_search[1] < 12:
        return None, None

    crop = frame.convert("L").crop(pixel_search)
    probe = crop.crop((0, 0, crop.width, max(1, crop.height // 3)))
    histogram = probe.histogram()
    halfway = max(1, sum(histogram)) / 2.0
    cumulative = 0
    background = 0
    for value, count in enumerate(histogram):
        cumulative += count
        if cumulative >= halfway:
            background = value
            break
    mask = crop.point(
        lambda value: 255 if abs(value - background) >= 28 else 0
    ).filter(ImageFilter.MaxFilter(3))
    pixels = mask.load()
    seen: set[tuple[int, int]] = set()
    components: list[tuple[int, int, int, int]] = []
    for y in range(crop.height):
        for x in range(crop.width):
            if not pixels[x, y] or (x, y) in seen:
                continue
            stack = [(x, y)]
            seen.add((x, y))
            xs: list[int] = []
            ys: list[int] = []
            while stack:
                current_x, current_y = stack.pop()
                xs.append(current_x)
                ys.append(current_y)
                for delta_y in (-1, 0, 1):
                    for delta_x in (-1, 0, 1):
                        neighbor_x = current_x + delta_x
                        neighbor_y = current_y + delta_y
                        neighbor = (neighbor_x, neighbor_y)
                        if (
                            0 <= neighbor_x < crop.width
                            and 0 <= neighbor_y < crop.height
                            and pixels[neighbor_x, neighbor_y]
                            and neighbor not in seen
                        ):
                            seen.add(neighbor)
                            stack.append(neighbor)
            component = (min(xs), min(ys), max(xs) + 1, max(ys) + 1)
            component_width = component[2] - component[0]
            component_height = component[3] - component[1]
            touches_edge = (
                component[0] <= 1
                or component[1] <= 1
                or component[2] >= crop.width - 1
                or component[3] >= crop.height - 1
            )
            if (
                len(xs) >= 10
                and 4 <= component_width <= crop.width * 0.45
                and 4 <= component_height <= crop.height * 0.70
                and not touches_edge
            ):
                components.append(
                    (
                        component[0] + pixel_search[0],
                        component[1] + pixel_search[1],
                        component[2] + pixel_search[0],
                        component[3] + pixel_search[1],
                    )
                )

    candidate_pixels = (
        candidate_bounds[0] * width / 1000.0,
        candidate_bounds[1] * height / 1000.0,
        candidate_bounds[2] * width / 1000.0,
        candidate_bounds[3] * height / 1000.0,
    )
    matches: list[tuple[int, int, int, int]] = []
    components = [
        component
        for component in components
        if not any(
            component is not other
            and other[0] <= component[0]
            and other[1] <= component[1]
            and other[2] >= component[2]
            and other[3] >= component[3]
            and (other[2] - other[0]) * (other[3] - other[1])
            > (component[2] - component[0]) * (component[3] - component[1])
            for other in components
        )
    ]
    if components:
        largest_component_area = max(
            (component[2] - component[0]) * (component[3] - component[1])
            for component in components
        )
        components = [
            component
            for component in components
            if (component[2] - component[0]) * (component[3] - component[1])
            >= largest_component_area * 0.40
        ]
    for component in components:
        component_area = max(1, (component[2] - component[0]) * (component[3] - component[1]))
        overlap = _bounds_intersection_area(
            tuple(float(part) for part in component),
            candidate_pixels,
        )
        if overlap / component_area >= 0.20:
            matches.append(component)
    component: tuple[int, int, int, int] | None = (
        matches[0] if len(matches) == 1 else None
    )
    component_assignments: dict[int, tuple[int, int, int, int]] = {}
    if (
        component is None
        and not matches
        and len(components) >= 2
        and len(components) == len(payload["controls"])
    ):
        control_bounds = [
            tuple(float(part) for part in control["bounds"])
            for control in payload["controls"]
        ]
        control_x_span = max(
            (bounds[0] + bounds[2]) / 2.0 for bounds in control_bounds
        ) - min((bounds[0] + bounds[2]) / 2.0 for bounds in control_bounds)
        control_y_span = max(
            (bounds[1] + bounds[3]) / 2.0 for bounds in control_bounds
        ) - min((bounds[1] + bounds[3]) / 2.0 for bounds in control_bounds)
        component_x_span = max(
            (bounds[0] + bounds[2]) / 2.0 for bounds in components
        ) - min((bounds[0] + bounds[2]) / 2.0 for bounds in components)
        component_y_span = max(
            (bounds[1] + bounds[3]) / 2.0 for bounds in components
        ) - min((bounds[1] + bounds[3]) / 2.0 for bounds in components)
        control_axis = "x" if control_x_span >= control_y_span else "y"
        component_axis = "x" if component_x_span >= component_y_span else "y"
        if (
            control_axis == component_axis
            and max(control_x_span, control_y_span) >= 20.0
            and max(component_x_span, component_y_span) >= 6.0
        ):
            axis = 0 if control_axis == "x" else 1
            ordered_controls = sorted(
                payload["controls"],
                key=lambda item: (
                    float(item["bounds"][axis])
                    + float(item["bounds"][axis + 2])
                )
                / 2.0,
            )
            ordered_components = sorted(
                components,
                key=lambda item: (item[axis] + item[axis + 2]) / 2.0,
            )
            candidate_index = next(
                index
                for index, control in enumerate(ordered_controls)
                if control is candidate
            )
            component = ordered_components[candidate_index]
            component_assignments = {
                id(control): ordered_components[index]
                for index, control in enumerate(ordered_controls)
            }
    if component is None:
        return None, None
    if not component_assignments:
        component_assignments[id(candidate)] = component

    def normalize_component(
        item: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        return (
            max(0, min(1000, round(item[0] * 1000 / width))),
            max(0, min(1000, round(item[1] * 1000 / height))),
            max(0, min(1000, round(item[2] * 1000 / width))),
            max(0, min(1000, round(item[3] * 1000 / height))),
        )

    normalized = normalize_component(component)
    if not _valid_1000_bounds(normalized):
        return None, None
    value = dict(payload)
    controls: list[dict[str, Any]] = []
    for control in payload["controls"]:
        item = dict(control)
        assigned = component_assignments.get(id(control))
        if assigned is not None:
            item["bounds"] = list(normalize_component(assigned))
        controls.append(item)
    value["controls"] = controls
    return value, normalized


def _is_literal_reload_element(item: dict[str, Any]) -> bool:
    terms = {
        "refresh",
        "reload",
        "refresh_page",
        "reload_page",
        "刷新",
        "重新加载",
        "刷新页面",
    }
    return (
        str(item.get("meaning") or "").strip().casefold() in terms
        or str(item.get("label") or "").strip().casefold() in terms
    )


def _apply_icon_cluster_audit(
    scene: UIScene,
    raw: str,
    *,
    fingerprint: str,
    localization_verified: bool,
    roi_bounds: tuple[int, int, int, int] | None,
) -> tuple[UIScene, int, bool]:
    """Mint one local reload candidate only from a strict visual cluster audit."""

    payload = _strict_icon_cluster_audit_payload(raw)
    controls = payload["controls"]
    reload_controls = [
        control for control in controls if control["semantic_class"] == "reload"
    ]
    candidate = reload_controls[0] if len(reload_controls) == 1 else None
    attested = bool(
        localization_verified
        and payload["cluster_complete"] is True
        and candidate is not None
        and float(candidate["confidence"]) >= 0.90
        and candidate["fully_visible"] is True
        and candidate["single_glyph"] is True
        and {"curved_arc", "arrowhead"}.issubset(candidate["shape_cues"])
        and not _ICON_CLUSTER_RELOAD_FORBIDDEN_CUES.intersection(
            candidate["shape_cues"]
        )
    )
    if attested:
        candidate_bounds = tuple(float(part) for part in candidate["bounds"])
        if roi_bounds is not None and not _bounds_inside(
            candidate_bounds,
            tuple(float(part) for part in roi_bounds),
            tolerance=20.0,
        ):
            attested = False
        if any(
            _bounds_intersection_area(
                candidate_bounds,
                tuple(float(part) for part in other["bounds"]),
            )
            > 0.0
            for other in controls
            if other is not candidate
        ):
            attested = False

    value = scene.to_dict()
    retained: list[dict[str, Any]] = []
    for item in value.get("elements") or []:
        if not isinstance(item, dict):
            continue
        if _is_literal_reload_element(item):
            # Model-authored reload claims never survive into an action scene.
            # The local audit below recreates at most one evidence-bound target.
            continue
        normalized = dict(item)
        states = dict(normalized.get("states") or {})
        states["goal_relevant"] = False
        normalized["states"] = states
        retained.append(normalized)
    if attested and candidate is not None:
        retained.append(
            {
                "element_id": "local_audited_reload_control_1",
                "role": "icon",
                "meaning": "reload",
                "label": "",
                "bounds": [float(part) / 1000.0 for part in candidate["bounds"]],
                "confidence": float(candidate["confidence"]),
                "states": {
                    "goal_relevant": True,
                    "fully_visible": True,
                    "reload_visual_audit": True,
                    "independent_geometry_verified": True,
                    "geometry_audit_source": "icon_cluster_localization",
                },
                "evidence": [
                    "严格图标簇审计确认完整圆弧、箭头头部且与相邻图标分离"
                ],
            }
        )
    value["elements"] = retained
    return (
        UIScene.from_dict(
            value,
            coordinate_scale=1.0,
            stable_override=True,
            fingerprint_override=fingerprint,
        ),
        len(controls),
        attested,
    )


def _goal_requests_input(context: dict[str, Any]) -> bool:
    focused = _active_subgoal_visual_context(context)
    if _goal_active_input_transaction_text(context):
        # A candidate-selection node may correctly describe only the visible
        # literal (for example, a Chinese IME candidate) without repeating
        # words such as "input field" or "keyboard".  The bridge mints this
        # marker only from a typed ``input.value_equals`` desired state. It is
        # a read-only audit trigger, not action authority: the dedicated input
        # audit must still prove one application input, the matching preedit
        # and one complete exact candidate before minting ``ime_exact_candidate``.
        return True
    if _goal_requests_keyboard_mode_switch(context):
        # A standalone deterministic tap on the keyboard mode switch has no
        # text payload, but it still requires the same independent whole-frame
        # input audit as an ordinary text transaction.  The mode helper keeps
        # the root-context exception limited to the exact-action wrapper.
        return True
    if focused is context:
        input_context: dict[str, Any] = context
    else:
        # Goal-wide entities remain available to the decision layer, but they
        # must not activate a future input audit while the current subgoal is
        # reload/navigation.  The current subgoal's own wording is the audit
        # trigger; this keeps independent target audits from overwriting one
        # another across graph nodes.
        input_context = {
            "objective": focused.get("objective"),
            "completion_conditions": focused.get("completion_conditions"),
        }
    visible = json.dumps(
        input_context,
        ensure_ascii=False,
    ).casefold()
    return any(
        term in visible
        for term in (
            "输入框",
            "文本框",
            "搜索框",
            "编辑框",
            "地址栏",
            "字段进入编辑",
            "字段获得焦点",
            "字段内容",
            "草稿区域",
            "草稿字段",
            "input field",
            "search box",
            "text field",
            "editable field",
            "draft field",
            "draft area",
            "address bar",
            "textbox",
            "input_text",
            "输入模式",
            "直输模式",
            "键盘模式",
            "软键盘",
            "input mode",
            "keyboard mode",
            "soft keyboard",
            " ime ",
            "direct_latin",
            "chinese_pinyin",
        )
    )


def _goal_requests_system_ui_audit(context: dict[str, Any]) -> bool:
    context = _active_subgoal_visual_context(context)
    evidence_selectors: dict[str, Any] = {
        "objective": context.get("objective"),
        "target_ui_label": context.get("target_ui_label"),
        "completion_conditions": context.get("completion_conditions"),
        "success_criteria": context.get("success_criteria"),
    }
    entities = context.get("entities")
    if isinstance(entities, dict):
        evidence_selectors["entity_target_ui_label"] = entities.get(
            "target_ui_label"
        )
    visible = json.dumps(evidence_selectors, ensure_ascii=False).casefold()
    return any(
        term in visible
        for term in (
            "系统导航栏",
            "导航栏",
            "系统手势",
            "全屏",
            "沉浸",
            "system navigation",
            "navigation bar",
            "system gesture",
            "fullscreen",
            "immersive",
        )
    )


def _goal_requests_reload(context: dict[str, Any]) -> bool:
    visible = json.dumps(
        _active_subgoal_visual_context(context),
        ensure_ascii=False,
    ).casefold()
    return any(
        marker in visible
        for marker in (
            "重新加载",
            "刷新",
            "刷新页面",
            "页面刷新",
            "reload",
            "refresh",
        )
    )


def _goal_requests_keyboard_mode_switch(context: dict[str, Any]) -> bool:
    focused = _active_subgoal_visual_context(context)
    selectors: list[Any] = [focused]
    entities = context.get("entities")
    focused_entities = focused.get("goal_entities")
    if (
        focused is not context
        and str(focused.get("subgoal_id") or "").strip()
        == "exact_tap_semantic"
        and isinstance(focused_entities, dict)
        and str(focused_entities.get("target_ui_label") or "").strip()
        and isinstance(entities, dict)
        and isinstance(entities.get("original_goal_visual_context"), str)
        and entities["original_goal_visual_context"].strip()
    ):
        # The deterministic exact-action bridge deliberately replaces the
        # active objective with generic tap wording.  Restore only its original
        # read-only visual intent so a keyboard-mode target can invoke the
        # dedicated audit.  Normal graph nodes keep the active-subgoal-only
        # boundary and cannot see future workflow text.
        selectors.append(entities["original_goal_visual_context"])
    visible = json.dumps(selectors, ensure_ascii=False).casefold()
    return any(
        term in visible
        for term in (
            "切换输入模式",
            "输入模式切换",
            "切换到英文",
            "切到英文",
            "英文直输",
            "切换到中文",
            "切到中文",
            "切换直输模式",
            "切换为直输模式",
            "switch input mode",
            "switch keyboard mode",
            "direct_latin",
            "chinese_pinyin",
        )
    )


def _goal_requests_local_text_clear(context: dict[str, Any]) -> bool:
    if not _goal_requests_input(context):
        return False
    visible = json.dumps(context, ensure_ascii=False).casefold()
    return any(
        term in visible
        for term in (
            "清空",
            "清除",
            "置空",
            "文字变为空",
            "内容变为空",
            "恢复为空",
            "恢复为空白",
            "clear text",
            "clear the text",
            "empty the input",
            "empty the field",
            "remove the text",
        )
    )


def _goal_requests_active_verified_text_clear(context: dict[str, Any]) -> bool:
    """Return true only when the current graph node is the clear step."""

    if not _goal_requests_input(context):
        return False
    focused = _active_subgoal_visual_context(context)
    visible = str(focused.get("objective") or "").casefold()
    return any(
        term in visible
        for term in (
            "清空",
            "清除",
            "置空",
            "删除",
            "文字变为空",
            "内容变为空",
            "恢复为空",
            "恢复为空白",
            "clear text",
            "clear the text",
            "clear draft",
            "empty the input",
            "empty the field",
            "remove the text",
            "delete",
        )
    )


def _should_audit_prefilled_input(scene: UIScene, context: dict[str, Any]) -> bool:
    if not _goal_requests_input(context):
        return False
    if _goal_requests_keyboard_mode_switch(context):
        return True
    trusted_inputs = tuple(
        item
        for item in scene.elements
        if item.role == "input"
        and item.states.get("goal_relevant") is True
        and float(item.confidence) >= 0.9
    )
    if len(trusted_inputs) != 1:
        return True
    states = trusted_inputs[0].states
    if not isinstance(states.get("value"), str):
        return True
    # A compact-pass input without explicit full-visibility evidence is not a
    # safe geometry source.  Run the independent input-structure audit even
    # before the keyboard is visible so a focus tap is bound to the complete
    # application field instead of a model-estimated box.
    if (
        states.get("fully_visible") is not True
        and not trusted_inputs[0].element_id.startswith("local_structured_input_")
    ):
        return True
    keyboard_is_relevant = states.get("focused") is True or _scene_reports_keyboard(
        scene
    )
    if not keyboard_is_relevant:
        return False
    # Compact observation may correctly see the field while misclassifying the
    # active IME mode. Any visible/focused keyboard on an input goal therefore
    # requires the independent whole-frame structure audit before planning.
    return True


def _input_audit_established_local_target(scene: UIScene) -> bool:
    """Return true only for authority minted by the dedicated input audit."""

    local_ids = {
        "local_audited_input_1",
        "local_audited_keyboard_mode_switch_1",
        "local_audited_ime_candidate_1",
        "local_audited_literal_key_1",
        "local_audited_enter_key_1",
        "local_audited_next_field_key_1",
        "local_audited_keyboard_layout_switch_1",
        "local_audited_keyboard_case_switch_1",
    }
    candidates = tuple(
        element
        for element in scene.elements
        if element.element_id in local_ids
        and element.states.get("goal_relevant") is True
        and element.states.get("fully_visible") is True
        and float(element.confidence) >= MIN_TARGET_CONFIDENCE
    )
    return len(candidates) == 1


def _goal_explicit_input_text(context: dict[str, Any]) -> str:
    """Return the active graph node's canonical text-entry payload, if any."""

    focused = _active_subgoal_visual_context(context)
    if focused is context:
        entities = context.get("entities")
    else:
        entities = focused.get("goal_entities")
    if not isinstance(entities, dict):
        return ""
    value = entities.get("input_text")
    return value.strip() if isinstance(value, str) else ""


def _goal_active_input_transaction_text(context: dict[str, Any]) -> str:
    """Return a bridge-minted typed input desired state for this active node."""

    focused = _active_subgoal_visual_context(context)
    if focused is context:
        return ""
    entities = focused.get("goal_entities")
    if not isinstance(entities, dict):
        return ""
    marker = entities.get("active_input_transaction_text")
    return marker if isinstance(marker, str) and marker else ""


def _goal_active_input_field(context: dict[str, Any]) -> tuple[str, str, bool]:
    """Return bridge-minted field identity, visible label and multiline flag."""

    focused = _active_subgoal_visual_context(context)
    if focused is context:
        return ("", "", False)
    entities = focused.get("goal_entities")
    if not isinstance(entities, dict):
        return ("", "", False)
    field_id = entities.get("active_input_field_id")
    field_label = entities.get("active_input_field_label", "")
    multiline = entities.get("active_input_multiline", False)
    if (
        not isinstance(field_id, str)
        or not field_id
        or not isinstance(field_label, str)
        or not isinstance(multiline, bool)
    ):
        return ("", "", False)
    return (field_id, field_label, multiline)


def _goal_has_unique_typed_active_input_field(context: dict[str, Any]) -> bool:
    """Return whether the active input marker exactly names one root typed field."""

    root_entities = context.get("entities")
    if not isinstance(root_entities, dict):
        return False
    fields = root_entities.get("input_fields")
    if not isinstance(fields, list):
        return False
    field_id, field_label, _multiline = _goal_active_input_field(context)
    text = _goal_active_input_transaction_text(context)
    if not field_id or not field_label or not text:
        return False
    exact_field_matches = sum(
        isinstance(item, dict)
        and item.get("field_id") == field_id
        and item.get("field_label") == field_label
        and item.get("text") == text
        for item in fields
    )
    exact_label_matches = sum(
        isinstance(item, dict)
        and item.get("field_label") == field_label
        for item in fields
    )
    return exact_field_matches == 1 and exact_label_matches == 1


def _goal_active_input_predecessor_field(
    context: dict[str, Any],
) -> tuple[str, str, str]:
    """Return one bridge-minted typed predecessor for a field transition."""

    focused = _active_subgoal_visual_context(context)
    if focused is context:
        return ("", "", "")
    goal_entities = focused.get("goal_entities")
    root_entities = context.get("entities")
    if not isinstance(goal_entities, dict) or not isinstance(root_entities, dict):
        return ("", "", "")
    field_id = goal_entities.get("active_input_predecessor_field_id")
    field_label = goal_entities.get("active_input_predecessor_field_label")
    text = goal_entities.get("active_input_predecessor_text")
    if not all(isinstance(value, str) and value for value in (field_id, field_label, text)):
        return ("", "", "")
    fields = root_entities.get("input_fields")
    active_field_id, active_field_label, _ = _goal_active_input_field(context)
    active_text = _goal_active_input_transaction_text(context)
    active_matches = [
        item for item in fields if isinstance(item, dict)
        and item.get("field_id") == active_field_id
        and item.get("field_label", "") == active_field_label
        and item.get("text") == active_text
    ] if isinstance(fields, list) else []
    matches = [
        item for item in fields if isinstance(item, dict)
        and item.get("field_id") == field_id
        and item.get("field_label") == field_label
        and item.get("text") == text
    ] if isinstance(fields, list) else []
    return (
        (field_id, field_label, text)
        if field_id != active_field_id
        and len(matches) == 1
        and len(active_matches) == 1
        else ("", "", "")
    )


def _goal_has_explicit_input_text(context: dict[str, Any]) -> bool:
    """Return true only when the graph supplied a concrete text-entry entity."""

    return bool(
        _goal_explicit_input_text(context)
        or _goal_active_input_transaction_text(context)
    )


def _goal_requests_keyboard_dismissal(context: dict[str, Any]) -> bool:
    visible = json.dumps(
        _active_subgoal_visual_context(context),
        ensure_ascii=False,
    ).casefold()
    keyboard = re.search(
        r"(?:软键盘|键盘|输入法|\b(?:soft\s+)?keyboard\b|\bime\b)",
        visible,
        re.IGNORECASE,
    )
    dismissal = re.search(
        r"(?:收起|隐藏|关闭|不再显示|不可见|未显示|"
        r"\b(?:hide|hidden|dismiss|close|closed|not\s+visible|no\s+longer\s+visible)\b)",
        visible,
        re.IGNORECASE,
    )
    return bool(keyboard and dismissal)


def _can_isolate_input_audit_from_attested_non_input(
    scene: UIScene,
    context: dict[str, Any],
) -> bool:
    """Keep a separately attested non-input target; consume no rejected audit."""

    if (
        not _goal_requests_reload(context)
        or _goal_requests_keyboard_mode_switch(context)
    ):
        return False
    candidate = scene.unique_trusted_goal_element()
    return bool(
        candidate is not None
        and candidate.role != "input"
        and candidate.meaning.casefold() == "reload"
        and candidate.states.get("reload_visual_audit") is True
        and candidate.states.get("fully_visible") is True
        and float(candidate.confidence) >= 0.9
        and any(item.strip() for item in candidate.evidence)
    )


def _scene_reports_keyboard(scene: UIScene) -> bool:
    visible = " ".join(
        [
            scene.summary,
            *scene.overlays,
            *(
                " ".join(
                    [
                        element.role,
                        element.meaning,
                        element.label,
                        *element.evidence,
                    ]
                )
                for element in scene.elements
            ),
        ]
    )
    return bool(
        re.search(r"(?:软键盘|输入法|键盘|keyboard|ime)", visible, re.IGNORECASE)
    )


def _normalized_keyboard_layout_token(value: Any) -> Any:
    """Normalize only exact, single-meaning aliases of formal layouts."""

    if not isinstance(value, str):
        return value
    normalized = value.strip().casefold()
    return {
        "symbols": "symbol",
        "symbol_grid": "symbol",
        "qwerty_symbol": "symbol",
    }.get(normalized, normalized)


def _evidenced_composite_symbol_layout(
    value: Any,
    *,
    keyboard: dict[str, Any],
    keyboard_bounds: tuple[float, float, float, float] | None,
    goal_context: dict[str, Any],
    current_input_text: str | None,
) -> str | None:
    """Normalize a composite numeric/symbol label from the next visible key.

    Composite names are descriptive model output, not a new layout enum.  A
    token is accepted only when every component is known, it explicitly names
    the symbol layer, and the current typed transaction independently derives
    one numeric or symbol character whose exact whole key is uniquely visible
    on this keyboard.  Letters and spaces remain qwerty-only and cannot turn a
    composite surface into input authority.
    """

    if not isinstance(value, str) or keyboard_bounds is None:
        return None
    components = tuple(
        part
        for part in re.split(r"[_+\-/\s]+", value.strip().casefold())
        if part
    )
    known_components = {
        "qwerty",
        "numeric",
        "number",
        "numbers",
        "symbol",
        "symbols",
        "grid",
        "layer",
    }
    if (
        len(components) < 2
        or not set(components).issubset(known_components)
        or not {"symbol", "symbols"}.intersection(components)
    ):
        return None
    literal_targets = set(
        _input_audit_literal_key_targets(
            _observation_goal_context(goal_context),
            current_input_text=current_input_text,
        )
    )
    if len(literal_targets) != 1:
        return None
    target = next(iter(literal_targets))
    if _preferred_keyboard_layout(target) not in {"numeric", "symbol"}:
        return None
    literal_keys = _validated_keyboard_literal_keys(
        keyboard.get("literal_keys", []),
        keyboard_bounds=keyboard_bounds,
    )
    if sum(item["value"] == target for item in literal_keys) != 1:
        return None
    return "symbol"


def _adjacent_exact_preedit_cue(
    trusted_input: dict[str, Any],
    trusted_preedits: list[dict[str, Any]],
    exact_text: str,
) -> bool:
    """Accept one exact non-authoritative cue beside the same input surface."""

    if (
        not exact_text
        or len(trusted_preedits) != 1
        or trusted_preedits[0].get("text") != exact_text
        or trusted_preedits[0].get("candidates")
    ):
        return False
    input_box = tuple(float(value) for value in trusted_input["input_bounds"])
    preedit_box = tuple(float(value) for value in trusted_preedits[0]["bounds"])
    horizontal_overlap = max(
        0.0,
        min(input_box[2], preedit_box[2]) - max(input_box[0], preedit_box[0]),
    )
    smaller_width = min(
        input_box[2] - input_box[0],
        preedit_box[2] - preedit_box[0],
    )
    vertical_gap = max(
        0.0,
        preedit_box[1] - input_box[3],
        input_box[1] - preedit_box[3],
    )
    return bool(
        smaller_width > 0
        and horizontal_overlap / smaller_width >= 0.60
        and vertical_gap <= 100
    )


def _resolve_pending_ime_candidate_input_state(
    trusted_input: dict[str, Any],
    trusted_preedits: list[dict[str, Any]],
    *,
    application_input_count: int,
    verified_input_lineage: TypedInputLineage | None,
    device_id: str,
    app_id: str,
    screen_id: str,
    input_field_id: str,
    authorized_text: str,
    keyboard_input_mode: str,
) -> dict[str, Any] | None:
    """Reduce one receipt-bound candidate result into the typed input ledger.

    The compact scene and its summary are deliberately absent from this
    reducer.  A pending exact-candidate lineage supplies the prior/expected
    transition; the dedicated audit supplies the unique field shell and the
    post-action full-field literal.  A prediction row may repeat the committed
    text, but it cannot retain a second preedit truth after this exact
    transition is proven.
    """

    if (
        application_input_count != 1
        or verified_input_lineage is None
        or verified_input_lineage.source
        != "pending_verified_ime_candidate_action"
        or keyboard_input_mode != "chinese_pinyin"
        or not input_field_id
        or input_field_id == "unknown"
        or input_field_id != verified_input_lineage.input_field_id
        or not isinstance(authorized_text, str)
        or not authorized_text.startswith(verified_input_lineage.exact_value)
        or trusted_input.get("right_button") is not None
        or not isinstance(trusted_input.get("caret_line_index"), int)
        or len(trusted_preedits) != 1
    ):
        return None
    exact_value = verified_input_lineage.exact_value
    raw_text = trusted_input.get("text")
    if not isinstance(raw_text, str) or raw_text not in {"", exact_value}:
        return None
    preedit = trusted_preedits[0]
    preedit_text = preedit.get("text")
    candidates = preedit.get("candidates")
    if (
        not isinstance(preedit_text, str)
        or not preedit_text
        or "\r" in preedit_text
        or "\n" in preedit_text
        or not exact_value.endswith(preedit_text)
        or not isinstance(candidates, list)
        or sum(
            isinstance(candidate, dict)
            and candidate.get("text") == preedit_text
            for candidate in candidates
        )
        != 1
    ):
        return None
    input_box = tuple(float(value) for value in trusted_input["input_bounds"])
    preedit_box = tuple(float(value) for value in preedit["bounds"])
    if (
        _bounds_overlap_ratio(input_box, preedit_box) < 0.90
        or _bounds_overlap_ratio(preedit_box, input_box) < 0.90
    ):
        return None
    normalized_bounds = tuple(value / 1000.0 for value in input_box)
    if not verified_input_lineage.matches_pending_input_state_surface(
        device_id=device_id,
        app_id=app_id,
        screen_id=screen_id,
        input_bounds=normalized_bounds,
        input_field_id=input_field_id,
    ):
        return None
    return {
        "committed_value": exact_value,
        "consumed_preedit": preedit,
    }


def _authorized_exact_committed_prefix_cue(
    trusted_input: dict[str, Any],
    trusted_preedits: list[dict[str, Any]],
    *,
    goal_context: dict[str, Any],
    keyboard_input_mode: str,
) -> str:
    """Recover one committed authorized prefix inside the typed field."""

    field_id, _field_label, _multiline = _goal_active_input_field(goal_context)
    authorized = _goal_active_input_transaction_text(goal_context)
    cues = trusted_input.get("visible_editable_cues")
    if (
        not field_id
        or field_id == "unknown"
        or not isinstance(authorized, str)
        or not authorized
        or keyboard_input_mode != "direct_latin"
        or trusted_input.get("text") != ""
        or not isinstance(cues, list)
    ):
        return ""
    non_text_cues = {
        "border",
        "caret",
        "cursor",
        "focus border",
        "focus ring",
        "outline",
    }
    literal_cues = tuple(
        item.strip()
        for item in cues
        if isinstance(item, str)
        and item.strip()
        and item.strip().casefold() not in non_text_cues
    )
    if len(literal_cues) != 1:
        return ""
    cue = literal_cues[0]
    if (
        not authorized.startswith(cue)
        or cue == trusted_input.get("placeholder")
        or cue in trusted_input.get("field_labels", ())
        or "\r" in cue
        or "\n" in cue
    ):
        return ""
    if trusted_preedits and not _adjacent_exact_preedit_cue(
        trusted_input,
        trusted_preedits,
        cue,
    ):
        return ""
    return cue


def _clear_goal_unique_committed_cue(
    trusted_input: dict[str, Any],
    trusted_preedits: list[dict[str, Any]],
    *,
    keyboard_input_mode: str,
) -> tuple[str, int]:
    """Recover visible committed glyphs only for an explicit clear-all goal.

    The extra visual-row count is not promoted to application text or a real
    newline.  It is only a conservative backspace unit for clearing the same
    focused field; an automatic soft wrap therefore cannot become content.
    """

    cues = trusted_input.get("visible_editable_cues")
    if (
        keyboard_input_mode != "direct_latin"
        or trusted_input.get("text") != ""
        or trusted_preedits
        or not isinstance(cues, list)
    ):
        return "", 0
    decorative = {
        "border",
        "caret",
        "cursor",
        "focus border",
        "focus ring",
        "outline",
        "|",
    }
    literals = tuple(
        item
        for item in cues
        if isinstance(item, str)
        and item.strip()
        and item.strip().casefold() not in decorative
        and "caret" not in item.casefold()
        and "cursor" not in item.casefold()
        and "光标" not in item
        and "插入符" not in item
    )
    if len(literals) != 1:
        return "", 0
    cue = literals[0]
    if (
        cue == trusted_input.get("placeholder")
        or cue in trusted_input.get("field_labels", ())
        or "\r" in cue
        or "\n" in cue
    ):
        return "", 0
    caret_line_index = trusted_input.get("caret_line_index")
    extra_rows = (
        caret_line_index
        if isinstance(caret_line_index, int)
        and not isinstance(caret_line_index, bool)
        and caret_line_index > 0
        else 0
    )
    return cue, extra_rows


def _locally_detect_caret_visual_row(
    trusted_input: dict[str, Any],
    *,
    frames: tuple[Image.Image, ...] | None,
    qwerty_anchors: dict[str, list[int]] | None,
    allow_goal_bound_local_detection: bool = False,
) -> int | None:
    """Locate a caret row with local calibrated pixels.

    A literal model caret cue normally gates the search.  An explicit clear or
    newline-structure goal may instead request the same bounded local search,
    because dropping visible pixels merely when the model omits the cue would
    make the exact structure unverifiable.  Local code searches only the field
    band above audited QWERTY rows, rejects edge borders, finds a thin vertical
    run longer than ordinary glyph strokes, and compares its center with the
    remaining text-row bands.
    """

    cues = trusted_input.get("visible_editable_cues")
    model_attests_caret = isinstance(cues, list) and any(
        isinstance(cue, str)
        and (
            cue.strip() == "|"
            or "caret" in cue.casefold()
            or "cursor" in cue.casefold()
            or "光标" in cue
            or "插入符" in cue
        )
        for cue in cues
    )
    if not model_attests_caret and not allow_goal_bound_local_detection:
        return None
    if (
        not frames
        or not isinstance(qwerty_anchors, dict)
        or not all(key in qwerty_anchors for key in ("q", "p", "a", "l"))
    ):
        return None
    input_bounds = trusted_input.get("input_bounds")
    if not _valid_1000_bounds(input_bounds):
        return None
    q_points = (qwerty_anchors["q"], qwerty_anchors["p"])
    if any(
        not isinstance(point, list)
        or len(point) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in point
        )
        for point in q_points
    ):
        return None
    q_row_y = sum(float(point[1]) for point in q_points) / len(q_points)
    left = max(0.0, float(input_bounds[0]))
    right = min(1000.0, float(input_bounds[2]))
    top = max(0.0, q_row_y - 240.0)
    bottom = min(1000.0, q_row_y - 80.0)
    if right - left < 120 or bottom - top < 80:
        return None

    detected_rows: list[int] = []
    for source in frames:
        if not isinstance(source, Image.Image) or source.width < 100 or source.height < 100:
            continue
        box = (
            round(left * source.width / 1000.0),
            round(top * source.height / 1000.0),
            round(right * source.width / 1000.0),
            round(bottom * source.height / 1000.0),
        )
        crop = source.convert("RGB").crop(box)
        width, height = crop.size
        if width < 40 or height < 40:
            continue
        pixels = crop.load()
        channel_values = [
            sorted(
                pixels[x, y][channel]
                for x in range(width)
                for y in range(height)
            )
            for channel in range(3)
        ]
        midpoint = width * height // 2
        background = tuple(values[midpoint] for values in channel_values)

        def foreground(x: int, y: int) -> bool:
            pixel = pixels[x, y]
            return bool(
                max(abs(pixel[index] - background[index]) for index in range(3))
                > 45
                or max(pixel) - min(pixel) > 55
            )

        candidates: list[tuple[int, int, int, int]] = []
        for x in range(round(width * 0.05), round(width * 0.95)):
            run_start: int | None = None
            previous: int | None = None
            for y in range(height):
                if not foreground(x, y):
                    continue
                if run_start is None or previous is None or y - previous > 2:
                    if run_start is not None and previous is not None:
                        candidates.append((previous - run_start + 1, x, run_start, previous))
                    run_start = y
                previous = y
            if run_start is not None and previous is not None:
                candidates.append((previous - run_start + 1, x, run_start, previous))
        if not candidates:
            continue
        run_length, caret_x, caret_top, caret_bottom = max(candidates)
        if run_length < max(18, round(height * 0.14)):
            continue
        near_columns = {
            x
            for length, x, start, end in candidates
            if abs(x - caret_x) <= 4
            and length >= 0.70 * run_length
            and abs(start - caret_top) <= 5
            and abs(end - caret_bottom) <= 5
        }
        if not 1 <= len(near_columns) <= 6:
            continue

        row_counts: list[int] = []
        for y in range(height):
            row_counts.append(
                sum(
                    foreground(x, y)
                    for x in range(width)
                    if abs(x - caret_x) > 5
                )
            )
        bands: list[tuple[int, int]] = []
        start: int | None = None
        previous: int | None = None
        for y, count in enumerate(row_counts):
            if count < 4:
                continue
            if start is None or previous is None or y - previous > 2:
                if start is not None and previous is not None:
                    bands.append((start, previous))
                start = y
            previous = y
        if start is not None and previous is not None:
            bands.append((start, previous))
        text_bands = [
            band
            for band in bands
            if band[0] > 3
            and band[1] < height - 4
            and band[1] - band[0] + 1 >= 4
        ]
        if not text_bands:
            continue
        caret_center = (caret_top + caret_bottom) / 2.0
        line_index: int | None = None
        for index, (band_top, band_bottom) in enumerate(text_bands):
            band_height = band_bottom - band_top + 1
            if band_top - 0.35 * band_height <= caret_center <= band_bottom + 0.35 * band_height:
                line_index = index
                break
            if caret_center < band_top:
                line_index = max(0, index - 1)
                break
        if line_index is None and caret_center > text_bands[-1][1]:
            line_index = len(text_bands)
        if line_index is not None and 0 <= line_index <= 30:
            detected_rows.append(line_index)

    if not detected_rows:
        return None
    counts = {row: detected_rows.count(row) for row in set(detected_rows)}
    best_row, best_count = max(counts.items(), key=lambda item: item[1])
    if len(detected_rows) > 1 and best_count < 2:
        return None
    return best_row


def _unique_clearable_ime_preedit(
    trusted_input: dict[str, Any],
    trusted_preedits: list[dict[str, Any]],
) -> str:
    """Bind one adjacent IME composition to one typed input for clearing."""

    if len(trusted_preedits) != 1:
        return ""
    preedit_text = trusted_preedits[0].get("text")
    if not isinstance(preedit_text, str) or not preedit_text:
        return ""
    input_box = tuple(float(value) for value in trusted_input["input_bounds"])
    preedit_box = tuple(float(value) for value in trusted_preedits[0]["bounds"])
    overlaps_input = bool(
        _bounds_overlap_ratio(input_box, preedit_box) > 0
        or _bounds_overlap_ratio(preedit_box, input_box) > 0
    )
    if overlaps_input:
        input_area = (input_box[2] - input_box[0]) * (
            input_box[3] - input_box[1]
        )
        preedit_area = (preedit_box[2] - preedit_box[0]) * (
            preedit_box[3] - preedit_box[1]
        )
        nested_empty_preedit = bool(
            trusted_input.get("text") == ""
            and _bounds_inside(preedit_box, input_box, tolerance=12)
            and _bounds_overlap_ratio(preedit_box, input_box) >= 0.90
            and input_area > 0
            and preedit_area <= 0.60 * input_area
        )
        return preedit_text if nested_empty_preedit else ""
    horizontal_overlap = max(
        0.0,
        min(input_box[2], preedit_box[2]) - max(input_box[0], preedit_box[0]),
    )
    smaller_width = min(
        input_box[2] - input_box[0],
        preedit_box[2] - preedit_box[0],
    )
    vertical_gap = max(
        0.0,
        preedit_box[1] - input_box[3],
        input_box[1] - preedit_box[3],
    )
    if (
        smaller_width <= 0
        or horizontal_overlap / smaller_width < 0.60
        or vertical_gap > 100
    ):
        return ""
    return preedit_text


def _unique_inline_ime_preedit_cue(
    trusted_input: dict[str, Any],
    trusted_preedits: list[dict[str, Any]],
    *,
    keyboard_input_mode: str,
) -> str:
    """Recover one literal pinyin composition rendered inside an empty field.

    Some applications draw the active IME composition over their editable
    surface while still reporting the committed application value as empty.
    The dedicated audit then returns that literal only as an editable cue,
    rather than as a separately bounded IME region.  Keep this recovery
    deliberately narrow: one empty input, one non-decorative pinyin literal,
    Chinese-pinyin mode, and no competing IME region.
    """

    cues = trusted_input.get("visible_editable_cues")
    if (
        keyboard_input_mode != "chinese_pinyin"
        or trusted_input.get("text") != ""
        or trusted_preedits
        or not isinstance(cues, list)
        or len(cues) != 1
    ):
        return ""
    cue = cues[0]
    if not isinstance(cue, str):
        return ""
    cue = cue.strip()
    non_text_cues = {
        "border",
        "caret",
        "cursor",
        "focus border",
        "focus ring",
        "outline",
    }
    if (
        not cue
        or cue.casefold() in non_text_cues
        or cue == trusted_input.get("placeholder")
        or cue in trusted_input.get("field_labels", ())
        or re.fullmatch(r"[a-z]+(?:'[a-z]+)*", cue, re.IGNORECASE) is None
    ):
        return ""
    return cue


def _next_field_key_element(
    key: dict[str, Any], *, source_field_id: str,
    target_field_id: str, target_field_label: str,
) -> dict[str, Any]:
    return {
        "element_id": "local_audited_next_field_key_1",
        "role": "button",
        "meaning": "input_next_field_key",
        "label": key["label"],
        "bounds": [part / 1000.0 for part in key["bounds"]],
        "confidence": key["confidence"],
        "states": {
            "goal_relevant": True, "fully_visible": True,
            "input_next_field_key": True, "key_action": "next",
            "source_input_field_id": source_field_id,
            "target_input_field_id": target_field_id,
            "target_input_field_label": target_field_label,
            "input_element_id": "local_audited_input_1",
        },
        "evidence": ["唯一聚焦typed字段与完整Next键绑定下一依赖字段"],
    }


def _apply_input_structure_audit(
    scene: UIScene,
    raw: str,
    *,
    fingerprint: str,
    goal_context: dict[str, Any],
    ledger_input_value: str | None = None,
    verified_input_lineage: TypedInputLineage | None = None,
    device_id: str | None = None,
    lineage_frame: Image.Image | None = None,
    qwerty_row_snapper: Callable[
        [list[Image.Image] | tuple[Image.Image, ...], dict[str, Any]],
        dict[str, list[int]] | None,
    ]
    | None = None,
    qwerty_row_frames: list[Image.Image] | tuple[Image.Image, ...] | None = None,
) -> UIScene:
    try:
        payload = _extract_json_object(raw)
        if qwerty_row_frames:
            _normalize_input_audit_pixel_coordinates(
                payload,
                frame_size=qwerty_row_frames[-1].size,
            )
        if set(payload) != {
            "protocol_version",
            "application_inputs",
            "ime_preedit_regions",
            "keyboard",
        }:
            raise UISceneError("输入结构审计包含协议外字段。")
        if payload.get("protocol_version") != INPUT_STRUCTURE_AUDIT_VERSION:
            raise UISceneError("输入结构审计协议版本不匹配。")
        application_inputs = payload.get("application_inputs")
        ime_preedit_regions = payload.get("ime_preedit_regions")
        keyboard = payload.get("keyboard")
        if not isinstance(application_inputs, list) or len(application_inputs) > 4:
            raise UISceneError("输入结构审计 application_inputs 必须是最多4项的数组。")
        if not isinstance(ime_preedit_regions, list) or len(ime_preedit_regions) > 4:
            raise UISceneError("输入结构审计 ime_preedit_regions 必须是最多4项的数组。")
        required_keyboard_fields = {
            "visible",
            "bounds",
            "layout",
            "input_mode",
            "mode_switch",
        }
        optional_keyboard_fields = {
            "qwerty_anchors",
            "backspace_key",
            "enter_key",
            "case_mode",
            "case_switch",
            "literal_keys",
            "layout_switches",
        }
        if (
            not isinstance(keyboard, dict)
            or not required_keyboard_fields.issubset(keyboard)
            or set(keyboard) - required_keyboard_fields - optional_keyboard_fields
        ):
            raise UISceneError("输入结构审计 keyboard 字段不符合协议。")

        keyboard_visible = keyboard.get("visible")
        keyboard_layout = keyboard.get("layout")
        if isinstance(keyboard_layout, str):
            normalized_layout = _normalized_keyboard_layout_token(
                keyboard_layout
            )
            if normalized_layout in {"qwerty", "numeric", "symbol", "unknown"}:
                keyboard_layout = normalized_layout
                keyboard["layout"] = normalized_layout
        keyboard_input_mode = keyboard.get("input_mode")
        keyboard_case_mode = keyboard.get("case_mode", "unknown")
        if not isinstance(keyboard_visible, bool):
            raise UISceneError("输入结构审计 keyboard.visible 必须是布尔值。")
        if keyboard_input_mode not in {
            "direct_latin",
            "chinese_pinyin",
            "unknown",
        }:
            raise UISceneError("输入结构审计 keyboard.input_mode 无效。")
        if keyboard_case_mode not in {"lower", "upper", "unknown"}:
            raise UISceneError("输入结构审计 keyboard.case_mode 无效。")
        raw_audited_qwerty_anchors = (
            keyboard.get("qwerty_anchors")
            if isinstance(keyboard.get("qwerty_anchors"), dict)
            else None
        )
        locally_snapped_qwerty_anchors: dict[str, list[int]] | None = None
        if (
            keyboard_visible is True
            and keyboard_layout == "qwerty"
            and callable(qwerty_row_snapper)
            and qwerty_row_frames is not None
            and isinstance(keyboard.get("qwerty_anchors"), dict)
        ):
            locally_snapped_qwerty_anchors = qwerty_row_snapper(
                qwerty_row_frames,
                keyboard["qwerty_anchors"],
            )
            if locally_snapped_qwerty_anchors is not None:
                _normalize_input_audit_portrait_grid_from_local_rows(
                    payload,
                    locally_snapped_anchors=locally_snapped_qwerty_anchors,
                )
                _reattach_input_audit_to_unique_scene_field(
                    scene,
                    application_inputs,
                    goal_context=goal_context,
                )
        keyboard_bounds: tuple[float, float, float, float] | None = None
        boundsless_keyboard_dismissal = False
        typed_prefix_verification_only = False
        if keyboard_visible:
            locally_rebuilt_keyboard_bounds = (
                _keyboard_bounds_from_locally_snapped_qwerty_anchors(
                    locally_snapped_qwerty_anchors
                )
                if locally_snapped_qwerty_anchors is not None
                else None
            )
            valid_keyboard_bounds = locally_rebuilt_keyboard_bounds is not None
            if valid_keyboard_bounds:
                keyboard_bounds = locally_rebuilt_keyboard_bounds
            else:
                valid_keyboard_bounds = _valid_1000_bounds(keyboard.get("bounds"))
            if valid_keyboard_bounds:
                if keyboard_bounds is None:
                    keyboard_bounds = tuple(float(value) for value in keyboard["bounds"])
                valid_keyboard_bounds = bool(
                    keyboard_bounds[2] - keyboard_bounds[0] >= 300
                    and keyboard_bounds[3] - keyboard_bounds[1] >= 180
                )
                if (
                    not valid_keyboard_bounds
                    and keyboard_bounds[2] - keyboard_bounds[0] >= 300
                    and locally_snapped_qwerty_anchors is not None
                ):
                    local_row_y = [
                        float(point[1])
                        for point in locally_snapped_qwerty_anchors.values()
                    ]
                    keyboard_bounds = (
                        keyboard_bounds[0],
                        min(keyboard_bounds[1], min(local_row_y)),
                        keyboard_bounds[2],
                        max(keyboard_bounds[3], max(local_row_y)),
                    )
                    valid_keyboard_bounds = bool(
                        keyboard_bounds[3] - keyboard_bounds[1] >= 180
                    )
            if not valid_keyboard_bounds:
                if _typed_prefix_input_survives_invalid_keyboard_geometry(
                    application_inputs,
                    goal_context,
                ):
                    typed_prefix_verification_only = True
                    keyboard_bounds = None
                    keyboard_layout = "unknown"
                    keyboard_input_mode = "unknown"
                    keyboard_case_mode = "unknown"
                    keyboard["mode_switch"] = None
                    keyboard["backspace_key"] = None
                    keyboard["enter_key"] = None
                    keyboard["case_switch"] = None
                    keyboard["literal_keys"] = []
                    keyboard["layout_switches"] = []
                    keyboard.pop("qwerty_anchors", None)
                    ime_preedit_regions = []
                elif not (
                    _goal_requests_keyboard_dismissal(goal_context)
                    and _scene_reports_keyboard(scene)
                ):
                    raise UISceneError("可见键盘必须提供有效 bounds。")
                else:
                    # A dismissal-only subgoal can safely authorize Android Back
                    # without touching the keyboard. Keep only dual-source
                    # presence/focus facts; discard all ungrounded keyboard
                    # geometry, mode and switch claims so text input remains
                    # impossible from this observation.
                    keyboard_bounds = None
                    boundsless_keyboard_dismissal = True
                    keyboard_layout = "unknown"
                    keyboard_input_mode = "unknown"
                    keyboard_case_mode = "unknown"
                    keyboard["mode_switch"] = None
                    keyboard["backspace_key"] = None
                    keyboard["enter_key"] = None
                    keyboard["case_switch"] = None
                    keyboard["literal_keys"] = []
                    keyboard["layout_switches"] = []
                    keyboard.pop("qwerty_anchors", None)
        elif (
            keyboard.get("bounds") is not None
            or keyboard.get("mode_switch") is not None
            or keyboard.get("backspace_key") is not None
            or keyboard.get("enter_key") is not None
            or keyboard.get("case_switch") is not None
            or keyboard.get("literal_keys") not in (None, [])
            or keyboard.get("layout_switches") not in (None, [])
        ):
            raise UISceneError("不可见键盘不能包含键位或切换控件。")
        application_keyboard_bounds = keyboard_bounds
        if _valid_1000_bounds(keyboard.get("bounds")):
            reported_keyboard_bounds = tuple(
                float(value) for value in keyboard["bounds"]
            )
            if (
                reported_keyboard_bounds[2] - reported_keyboard_bounds[0] >= 300
            ):
                # A same-frame model outer rectangle may remain useful only to
                # separate App fields from the IME surface.  Execution geometry
                # still comes from the independently snapped local rows above.
                application_keyboard_bounds = reported_keyboard_bounds
        if keyboard_layout not in {"qwerty", "numeric", "symbol", "unknown"}:
            keyboard_layout = _evidenced_composite_symbol_layout(
                keyboard_layout,
                keyboard=keyboard,
                keyboard_bounds=keyboard_bounds,
                goal_context=goal_context,
                current_input_text=ledger_input_value,
            )
            if keyboard_layout is None:
                raise UISceneError("输入结构审计 keyboard.layout 无效。")
            keyboard["layout"] = keyboard_layout

        trusted_preedits: list[dict[str, Any]] = []
        for item in ime_preedit_regions:
            if not isinstance(item, dict) or set(item) not in (
                {"region_id", "bounds", "text", "confidence"},
                {"region_id", "bounds", "text", "confidence", "candidates"},
            ):
                raise UISceneError("IME预编辑区字段不符合协议。")
            if not _valid_1000_bounds(item.get("bounds")):
                raise UISceneError("IME预编辑区 bounds 不符合0..1000协议。")
            confidence = _audit_confidence(item.get("confidence"), "IME预编辑区")
            bounds = tuple(float(value) for value in item["bounds"])
            raw_candidates = item.get("candidates", [])
            if not isinstance(raw_candidates, list) or len(raw_candidates) > 8:
                raise UISceneError("IME候选必须是最多8项的数组。")
            candidates: list[dict[str, Any]] = []
            candidate_locations: list[tuple[str, float, float]] = []
            for candidate in raw_candidates:
                if not isinstance(candidate, dict) or set(candidate) != {
                    "text", "bounds", "confidence", "fully_visible"
                }:
                    raise UISceneError("IME候选字段不符合协议。")
                candidate_text = str(candidate.get("text") or "").strip()
                if not _valid_1000_bounds(candidate.get("bounds")):
                    raise UISceneError("IME候选 bounds 无效。")
                candidate_bounds = tuple(float(value) for value in candidate["bounds"])
                candidate_confidence = _audit_confidence(
                    candidate.get("confidence"), "IME候选"
                )
                if not isinstance(candidate.get("fully_visible"), bool):
                    raise UISceneError("IME候选 fully_visible 必须是布尔值。")
                candidate_vertical_gap = max(
                    0.0,
                    candidate_bounds[1] - bounds[3],
                    bounds[1] - candidate_bounds[3],
                )
                adjacent_to_preedit = bool(
                    _bounds_inside(candidate_bounds, bounds, tolerance=12)
                    or candidate_vertical_gap <= 120
                )
                keyboard_candidate_strip = False
                candidate_anchors = (
                    locally_snapped_qwerty_anchors
                    or raw_audited_qwerty_anchors
                )
                if (
                    not adjacent_to_preedit
                    and keyboard_visible
                    and keyboard_layout == "qwerty"
                    and application_keyboard_bounds is not None
                    and isinstance(candidate_anchors, dict)
                ):
                    try:
                        qwerty_top_row_y = statistics.mean(
                            (
                                float(candidate_anchors["q"][1]),
                                float(candidate_anchors["p"][1]),
                            )
                        )
                    except (KeyError, TypeError, ValueError):
                        qwerty_top_row_y = math.nan
                    candidate_height = candidate_bounds[3] - candidate_bounds[1]
                    keyboard_candidate_strip = bool(
                        math.isfinite(qwerty_top_row_y)
                        and 15 <= candidate_height <= 100
                        and _bounds_inside(
                            candidate_bounds,
                            application_keyboard_bounds,
                            tolerance=12,
                        )
                        and candidate_bounds[3] <= qwerty_top_row_y - 8
                        and candidate_bounds[1]
                        <= application_keyboard_bounds[1]
                        + min(
                            180.0,
                            0.35
                            * (
                                application_keyboard_bounds[3]
                                - application_keyboard_bounds[1]
                            ),
                        )
                    )
                if not adjacent_to_preedit and not keyboard_candidate_strip:
                    raise UISceneError("IME候选必须位于对应预编辑区内或紧邻候选行。")
                candidate_locations.append(
                    (
                        "keyboard_strip"
                        if keyboard_candidate_strip
                        else "preedit_adjacent",
                        (candidate_bounds[1] + candidate_bounds[3]) / 2.0,
                        candidate_bounds[3] - candidate_bounds[1],
                    )
                )
                # Candidate text is optional action authority.  A model may
                # enumerate a visible but irrelevant long preedit string here.
                # After its schema and geometry are validated, discard that
                # non-authoritative item instead of letting it veto a separately
                # grounded application input.  It can never become an action.
                if not candidate_text or len(candidate_text) > 20:
                    continue
                if candidate["fully_visible"] and candidate_confidence >= 0.9:
                    candidates.append(
                        {
                            "text": candidate_text,
                            "bounds": candidate_bounds,
                            "confidence": candidate_confidence,
                        }
                    )
            location_kinds = {item[0] for item in candidate_locations}
            if len(location_kinds) > 1:
                raise UISceneError("IME候选不能分散在预编辑附近和键盘候选栏。")
            if location_kinds == {"keyboard_strip"} and candidate_locations:
                centers = [item[1] for item in candidate_locations]
                heights = [item[2] for item in candidate_locations]
                if max(centers) - min(centers) > max(
                    24.0,
                    0.75 * statistics.median(heights),
                ):
                    raise UISceneError("键盘顶部IME候选必须位于同一水平候选行。")
            if confidence >= 0.9:
                trusted_preedits.append(
                    {
                        "text": str(item.get("text") or "").strip(),
                        "bounds": bounds,
                        "confidence": confidence,
                        "candidates": candidates,
                    }
                )

        active_field_id, active_field_label, active_multiline = (
            _goal_active_input_field(goal_context)
        )
        active_transaction_text = _goal_active_input_transaction_text(goal_context)
        explicit_input_text = _goal_explicit_input_text(goal_context)
        multiline_input_contract = bool(
            active_multiline
            or "\n" in explicit_input_text
            or "\r" in explicit_input_text
        )
        unique_typed_active_field = _goal_has_unique_typed_active_input_field(
            goal_context
        )
        active_label_occurrences = sum(
            sum(
                isinstance(label, str)
                and label.strip().casefold() == active_field_label.casefold()
                for label in item.get("field_labels", [])
            )
            for item in application_inputs
            if active_field_label
            and isinstance(item, dict)
            and isinstance(item.get("field_labels", []), list)
        )
        tall_typed_active_contract = bool(
            unique_typed_active_field and active_label_occurrences == 1
        )
        matches: list[dict[str, Any]] = []
        for item in application_inputs:
            required_input_fields = {
                "structure_id",
                "bounds",
                "fully_visible",
                "text",
                "placeholder",
                "visible_editable_cues",
                "caret_line_index",
                "confidence",
                "right_button",
            }
            if (
                not isinstance(item, dict)
                or not required_input_fields.issubset(item)
                or set(item) - required_input_fields - {"field_labels"}
            ):
                raise UISceneError("应用输入结构字段不符合协议。")
            button = item.get("right_button")
            if _can_discard_separate_right_button(item, button):
                # A separate adjacent control grants no authority. Discard
                # only that optional object; never widen or move the input.
                item["right_button"] = None
                button = None
            if button is not None and (
                not isinstance(button, dict)
                or set(button) != {"label", "bounds", "confidence"}
            ):
                raise UISceneError("输入结构审计 right_button 字段不符合协议。")
            if not isinstance(item.get("fully_visible"), bool):
                raise UISceneError("输入结构审计 fully_visible 必须是布尔值。")
            if not _valid_1000_bounds(item.get("bounds")):
                raise UISceneError("输入结构审计 bounds 不符合0..1000协议。")
            confidence = _audit_confidence(item.get("confidence"), "应用输入结构")
            cues = item.get("visible_editable_cues")
            if (
                not isinstance(cues, list)
                or len(cues) > 4
                or any(not isinstance(value, str) for value in cues)
            ):
                raise UISceneError("visible_editable_cues 必须是最多4项的字符串数组。")
            cues = [value.strip()[:120] for value in cues if value.strip()]
            caret_line_index = item.get("caret_line_index")
            if (
                caret_line_index is not None
                and (
                    isinstance(caret_line_index, bool)
                    or not isinstance(caret_line_index, int)
                    or not 0 <= caret_line_index <= 30
                )
            ):
                raise UISceneError(
                    "caret_line_index 必须是0..30整数或 null。"
                )
            raw_field_labels = item.get("field_labels", [])
            if (
                not isinstance(raw_field_labels, list)
                or len(raw_field_labels) > 6
                or any(
                    not isinstance(value, str)
                    or not value.strip()
                    or len(value.strip()) > 120
                    or "\n" in value
                    or "\r" in value
                    for value in raw_field_labels
                )
            ):
                raise UISceneError("field_labels 必须是最多6项的非空可见字符串数组。")
            field_labels = tuple(
                dict.fromkeys(value.strip() for value in raw_field_labels)
            )
            if not item["fully_visible"] or confidence < 0.9:
                continue
            raw_text = item.get("text")
            if not isinstance(raw_text, str):
                raise UISceneError("应用输入结构 text 必须是字符串。")
            # Exact input values may intentionally begin or end with spaces or
            # real line feeds.  Placeholder/cue strings are descriptive, but
            # the value itself must never be trimmed.
            text = raw_text
            placeholder = str(item.get("placeholder") or "").strip()
            pending_candidate_shell = bool(
                verified_input_lineage is not None
                and verified_input_lineage.source
                == "pending_verified_ime_candidate_action"
                and active_field_id not in {"", "unknown"}
            )
            if (
                not text
                and not placeholder
                and not cues
                and not pending_candidate_shell
            ):
                continue
            bounds = tuple(float(value) for value in item["bounds"])
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            pending_ime_candidate_state = _resolve_pending_ime_candidate_input_state(
                {
                    "text": text,
                    "placeholder": placeholder,
                    "visible_editable_cues": cues,
                    "caret_line_index": caret_line_index,
                    "field_labels": field_labels,
                    "input_bounds": [round(value) for value in bounds],
                    "right_button": button,
                },
                trusted_preedits,
                application_input_count=len(application_inputs),
                verified_input_lineage=verified_input_lineage,
                device_id=str(device_id or ""),
                app_id=scene.app_id,
                screen_id=scene.screen_id,
                input_field_id=active_field_id,
                authorized_text=active_transaction_text,
                keyboard_input_mode=keyboard_input_mode,
            )
            exact_active_label = bool(
                active_field_label
                and sum(
                    label.casefold() == active_field_label.casefold()
                    for label in field_labels
                )
                == 1
            )
            maximum_input_height = (
                600
                if multiline_input_contract
                or (tall_typed_active_contract and exact_active_label)
                else 180
            )
            if (
                bounds[1] <= 10
                or bounds[3] >= 990
                or width < 240
                or not 20 <= height <= maximum_input_height
                or (
                    application_keyboard_bounds is not None
                    and _bounds_overlap_ratio(
                        bounds,
                        application_keyboard_bounds,
                    )
                    >= 0.25
                )
                or any(
                    (
                        _bounds_overlap_ratio(bounds, preedit["bounds"]) >= 0.35
                        or _bounds_overlap_ratio(preedit["bounds"], bounds) >= 0.35
                    )
                    and (
                        pending_ime_candidate_state is None
                        or preedit
                        is not pending_ime_candidate_state["consumed_preedit"]
                    )
                    and not (
                        text == ""
                        and preedit["text"]
                        and _bounds_inside(
                            preedit["bounds"], bounds, tolerance=12
                        )
                        and _bounds_overlap_ratio(
                            preedit["bounds"], bounds
                        ) >= 0.90
                        and (
                            (preedit["bounds"][2] - preedit["bounds"][0])
                            * (preedit["bounds"][3] - preedit["bounds"][1])
                        )
                        <= 0.60 * width * height
                    )
                    for preedit in trusted_preedits
                )
            ):
                continue
            input_bounds = [round(value) for value in bounds]
            button_match: dict[str, Any] | None = None
            if button is not None:
                if not _valid_1000_bounds(button.get("bounds")):
                    raise UISceneError("输入结构审计 right_button bounds 无效。")
                button_confidence = _audit_confidence(
                    button.get("confidence"), "输入结构审计 right_button"
                )
                button_label = str(button.get("label") or "").strip()
                button_bounds = tuple(float(value) for value in button["bounds"])
                if (
                    button_confidence < 0.9
                    or not button_label
                    or not _bounds_inside(button_bounds, bounds, tolerance=20)
                    or button_bounds[0] <= bounds[0] + 0.55 * width
                    or _vertical_overlap_ratio(button_bounds, bounds) < 0.8
                ):
                    continue
                input_bounds[2] = round(button_bounds[0])
                button_match = {
                    "label": button_label,
                    "bounds": [round(value) for value in button_bounds],
                    "confidence": button_confidence,
                }
            if input_bounds[2] - input_bounds[0] < 120:
                continue
            matches.append(
                {
                    "text": text,
                    "placeholder": placeholder,
                    "visible_editable_cues": cues,
                    "caret_line_index": caret_line_index,
                    "field_labels": field_labels,
                    "input_bounds": input_bounds,
                    "right_button": button_match,
                    "pending_ime_candidate_state": pending_ime_candidate_state,
                    "confidence": min(
                        confidence,
                        float(button_match["confidence"])
                        if button_match is not None
                        else confidence,
                    ),
                }
            )

        switch_is_goal = _goal_requests_keyboard_mode_switch(goal_context)
        active_clear_goal = _goal_requests_active_verified_text_clear(goal_context)
        if active_field_label:
            field_matches = [
                item
                for item in matches
                if sum(
                    label.casefold() == active_field_label.casefold()
                    for label in item["field_labels"]
                )
                == 1
            ]
            trusted_input = field_matches[0] if len(field_matches) == 1 else None
        else:
            trusted_input = matches[0] if len(matches) == 1 else None
        if (
            trusted_input is not None
            and trusted_input.get("pending_ime_candidate_state") is not None
            and verified_input_lineage is not None
        ):
            resolved_ime_state = trusted_input["pending_ime_candidate_state"]
            committed_preedit = resolved_ime_state["consumed_preedit"]
            trusted_input = dict(trusted_input)
            trusted_input["lineage_ime_candidate_committed_value"] = (
                resolved_ime_state["committed_value"]
            )
            trusted_input["text"] = resolved_ime_state["committed_value"]
            trusted_preedits = [
                item for item in trusted_preedits if item is not committed_preedit
            ]
        predecessor_field_id, predecessor_field_label, predecessor_text = (
            _goal_active_input_predecessor_field(goal_context)
        )
        predecessor_audit_matches = [
            item for item in matches
            if sum(
                label.casefold() == predecessor_field_label.casefold()
                for label in item["field_labels"]
            ) == 1
            and item["text"] == predecessor_text
        ] if predecessor_field_id else []
        # The fresh dedicated audit is the sole visual value authority for the
        # predecessor.  A compact-scene transcription cannot restore or veto
        # it; the typed dependency supplies identity while the audit supplies
        # the exact committed value and the next-key geometry.
        predecessor_input = (
            dict(predecessor_audit_matches[0])
            if len(predecessor_audit_matches) == 1
            else None
        )
        if (
            trusted_input is not None
            and trusted_input.get("caret_line_index") is None
        ):
            local_caret_line_index = _locally_detect_caret_visual_row(
                trusted_input,
                frames=(
                    tuple(qwerty_row_frames)
                    if qwerty_row_frames is not None
                    else None
                ),
                # Raw audited anchors may define only the pixel-search band.
                # They never become action geometry unless the independent
                # multi-frame row snap above also succeeds.
                qwerty_anchors=(
                    locally_snapped_qwerty_anchors
                    or raw_audited_qwerty_anchors
                ),
                allow_goal_bound_local_detection=bool(
                    active_clear_goal
                    or (
                        isinstance(active_transaction_text, str)
                        and "\n" in active_transaction_text
                    )
                ),
            )
            if local_caret_line_index is not None:
                trusted_input = dict(trusted_input)
                trusted_input["caret_line_index"] = local_caret_line_index
                trusted_input["local_caret_line_index"] = local_caret_line_index
        if (
            trusted_input is not None
            and active_clear_goal
            and not trusted_input["text"]
        ):
            clear_cue, extra_clear_units = _clear_goal_unique_committed_cue(
                trusted_input,
                trusted_preedits,
                keyboard_input_mode=keyboard_input_mode,
            )
            if clear_cue:
                trusted_input = dict(trusted_input)
                trusted_input["clear_goal_visible_cue_text"] = clear_cue
                trusted_input["clear_extra_delete_units"] = extra_clear_units
                trusted_input["text"] = clear_cue
        if (
            trusted_input is not None
            and active_clear_goal
            and isinstance(trusted_input.get("text"), str)
            and trusted_input["text"]
            and isinstance(trusted_input.get("caret_line_index"), int)
        ):
            extra_clear_units = max(
                0,
                trusted_input["caret_line_index"]
                - trusted_input["text"].count("\n"),
            )
            if extra_clear_units > 0:
                trusted_input = dict(trusted_input)
                trusted_input["clear_extra_delete_units"] = extra_clear_units
        if (
            trusted_input is not None
            and not trusted_input["text"]
            and verified_input_lineage is None
        ):
            authorized_prefix_cue = _authorized_exact_committed_prefix_cue(
                trusted_input,
                trusted_preedits,
                goal_context=goal_context,
                keyboard_input_mode=keyboard_input_mode,
            )
            if authorized_prefix_cue:
                trusted_input = dict(trusted_input)
                trusted_input["authorized_prefix_visible_cue_text"] = (
                    authorized_prefix_cue
                )
                trusted_input["text"] = authorized_prefix_cue
        if trusted_input is not None and verified_input_lineage is not None:
            raw_lineage_text = trusted_input["text"]
            lineage_bounds = tuple(
                float(part) / 1000.0 for part in trusted_input["input_bounds"]
            )
            lineage_visible_cues = tuple(trusted_input["visible_editable_cues"])
            if (
                keyboard_input_mode == "direct_latin"
                and verified_input_lineage.exact_value not in lineage_visible_cues
                and _adjacent_exact_preedit_cue(
                    trusted_input,
                    trusted_preedits,
                    verified_input_lineage.exact_value,
                )
            ):
                lineage_visible_cues = (
                    *lineage_visible_cues,
                    verified_input_lineage.exact_value,
                )
            if verified_input_lineage.matches_trailing_newline_cue(
                device_id=str(device_id or ""),
                app_id=scene.app_id,
                screen_id=scene.screen_id,
                raw_value=raw_lineage_text,
                visible_editable_cues=lineage_visible_cues,
                caret_line_index=trusted_input.get("caret_line_index"),
                input_bounds=lineage_bounds,
                input_field_id=active_field_id,
                current_frame=lineage_frame,
            ):
                trusted_input = dict(trusted_input)
                trusted_input["verified_trailing_newline"] = True
                trusted_input["text"] = verified_input_lineage.exact_value
            elif verified_input_lineage.matches_pending_input_state_cue(
                device_id=str(device_id or ""),
                app_id=scene.app_id,
                screen_id=scene.screen_id,
                raw_value=raw_lineage_text,
                visible_editable_cues=lineage_visible_cues,
                input_bounds=lineage_bounds,
                input_field_id=active_field_id,
            ):
                trusted_input = dict(trusted_input)
                trusted_input["lineage_visible_cue_text"] = (
                    verified_input_lineage.exact_value
                )
                trusted_input["text"] = verified_input_lineage.exact_value
            elif verified_input_lineage.matches_persisted_surface_cue(
                device_id=str(device_id or ""),
                app_id=scene.app_id,
                screen_id=scene.screen_id,
                raw_value=raw_lineage_text,
                visible_editable_cues=lineage_visible_cues,
                input_bounds=lineage_bounds,
                current_frame=lineage_frame,
            ):
                trusted_input = dict(trusted_input)
                trusted_input["lineage_persisted_visible_cue_text"] = (
                    verified_input_lineage.exact_value
                )
                trusted_input["text"] = verified_input_lineage.exact_value
            elif (
                not _goal_active_input_field(goal_context)[2]
                and verified_input_lineage.matches_visual(
                device_id=str(device_id or ""),
                app_id=scene.app_id,
                screen_id=scene.screen_id,
                raw_value=raw_lineage_text,
                input_bounds=lineage_bounds,
                current_frame=lineage_frame,
                )
            ):
                trusted_input = dict(trusted_input)
                trusted_input["lineage_visual_text"] = raw_lineage_text
                trusted_input["text"] = verified_input_lineage.exact_value
            if trusted_input["text"] == "":
                pending_preedit_text = _unique_clearable_ime_preedit(
                    trusted_input,
                    trusted_preedits,
                )
                pending_committed_prefix = (
                    verified_input_lineage.pending_text_committed_prefix(
                        device_id=str(device_id or ""),
                        app_id=scene.app_id,
                        screen_id=scene.screen_id,
                        authorized_text=active_transaction_text,
                        raw_value=trusted_input["text"],
                        preedit_text=pending_preedit_text,
                        input_bounds=lineage_bounds,
                        input_field_id=active_field_id,
                    )
                )
                if pending_committed_prefix is not None:
                    trusted_input = dict(trusted_input)
                    trusted_input["lineage_pending_text_prefix"] = (
                        pending_committed_prefix
                    )
                    trusted_input["text"] = pending_committed_prefix
        exact_ime_candidate: dict[str, Any] | None = None
        exact_ime_preedit_text = ""
        input_step = None
        if (
            trusted_input is not None
            and _goal_has_explicit_input_text(goal_context)
            and not active_clear_goal
        ):
            focused_context = _active_subgoal_visual_context(goal_context)
            entities = (
                focused_context.get("goal_entities")
                if focused_context is not goal_context
                else goal_context.get("entities")
            )
            target_text = (
                _goal_active_input_transaction_text(goal_context)
                or (
                    entities.get("input_text")
                    if isinstance(entities, dict)
                    else None
                )
            )
            try:
                input_step = plan_next_verified_input(target_text, trusted_input["text"])
            except (ValueError, VerifiedTextTransactionError):
                input_step = None
            if input_step is not None and input_step.kind == "chinese_pinyin":
                matching_preedits = [
                    item
                    for item in trusted_preedits
                    if re.sub(r"[^a-z]", "", item["text"].casefold())
                    == input_step.pinyin
                ]
                matching_candidates = [
                    candidate
                    for item in matching_preedits
                    for candidate in item["candidates"]
                    if candidate["text"] == input_step.segment
                ]
                if len(matching_preedits) == 1 and len(matching_candidates) == 1:
                    exact_ime_candidate = matching_candidates[0]
                    # IMEs may insert a visible syllable separator (for
                    # example ``ni'hao``).  Candidate matching above already
                    # proves the same deterministic local pinyin.  Publish the
                    # canonical pinyin state so the typed transition and
                    # controller verify one authority instead of raw IME
                    # decoration.
                    exact_ime_preedit_text = input_step.pinyin
                elif len(matching_preedits) == 1:
                    raise UISceneError(
                        "有用输入法预编辑必须提供唯一逐字候选几何，不能转为清除。"
                    )
            elif input_step is not None and input_step.kind == "direct_latin":
                matching_preedits = [
                    item
                    for item in trusted_preedits
                    if item["text"] == input_step.segment
                ]
                matching_candidates = [
                    candidate
                    for item in matching_preedits
                    for candidate in item["candidates"]
                    if candidate["text"] == input_step.segment
                ]
                if len(matching_preedits) == 1 and len(matching_candidates) == 1:
                    exact_ime_candidate = matching_candidates[0]
                    exact_ime_preedit_text = matching_preedits[0]["text"]
                elif len(matching_preedits) == 1:
                    raise UISceneError(
                        "有用输入法预编辑必须提供唯一逐字候选几何，不能转为清除。"
                    )
        qwerty_geometry: dict[str, Any] | None = None
        raw_qwerty_anchors = keyboard.get("qwerty_anchors")
        if raw_qwerty_anchors is not None:
            if (
                not keyboard_visible
                or keyboard_layout != "qwerty"
                or keyboard_bounds is None
            ):
                raise UISceneError("QWERTY anchors 必须绑定完整可见的 QWERTY 键盘。")
            qwerty_geometry = _validated_qwerty_keyboard_geometry(
                locally_snapped_qwerty_anchors or raw_qwerty_anchors,
                keyboard_bounds=keyboard_bounds,
                locally_snapped=locally_snapped_qwerty_anchors is not None,
            )
        raw_backspace_key = keyboard.get("backspace_key")
        if (
            trusted_input is not None
            and isinstance(raw_backspace_key, dict)
            and set(raw_backspace_key) == {
                "label", "bounds", "confidence", "fully_visible"
            }
            and not str(raw_backspace_key.get("label") or "").strip()
        ):
            # An empty literal cannot identify or authorize a delete key. Drop
            # only this optional non-authoritative substructure so an otherwise
            # valid application input can still prove its post-action value.
            raw_backspace_key = None
        generic_backspace_geometry = _validated_keyboard_backspace_key(
            raw_backspace_key,
            keyboard_bounds=keyboard_bounds,
        )
        clearable_ime_preedit = ""
        if (
            trusted_input is not None
            and active_field_id
            and active_transaction_text
            and keyboard_visible
            and keyboard_bounds is not None
            and (
                qwerty_geometry is not None
                or generic_backspace_geometry is not None
            )
        ):
            clearable_ime_preedit = _unique_clearable_ime_preedit(
                trusted_input,
                trusted_preedits,
            )
            if not clearable_ime_preedit:
                clearable_ime_preedit = _unique_inline_ime_preedit_cue(
                    trusted_input,
                    trusted_preedits,
                    keyboard_input_mode=keyboard_input_mode,
                )
        enter_key = None
        if locally_snapped_qwerty_anchors is not None:
            enter_key = _locally_snapped_keyboard_enter_key(
                keyboard.get("enter_key"),
                anchors=locally_snapped_qwerty_anchors,
                keyboard_bounds=keyboard_bounds,
            )
        else:
            enter_key = _validated_keyboard_enter_key(
                keyboard.get("enter_key"),
                keyboard_bounds=keyboard_bounds,
            )
        next_field_key = (
            enter_key
            if predecessor_input is not None
            and enter_key is not None
            and enter_key["key_action"] == "next"
            else None
        )
        raw_mode_switch = keyboard.get("mode_switch")
        input_needs_mode_switch = bool(
            not active_clear_goal
            and not clearable_ime_preedit
            and exact_ime_candidate is None
            and input_step is not None
            and input_step.kind in {"direct_latin", "chinese_pinyin"}
            and keyboard_visible
            and keyboard_layout == "qwerty"
            and keyboard_input_mode in {"direct_latin", "chinese_pinyin"}
            and keyboard_input_mode != input_step.required_mode
        )
        switch_is_goal = switch_is_goal or input_needs_mode_switch
        discardable_mode_switch_geometry = bool(
            isinstance(raw_mode_switch, dict)
            and set(raw_mode_switch)
            == {"label", "bounds", "confidence", "current_mode", "target_mode"}
            and not _valid_1000_bounds(raw_mode_switch.get("bounds"))
        )
        discardable_mode_switch_semantic_conflict = False
        if (
            isinstance(raw_mode_switch, dict)
            and set(raw_mode_switch)
            == {"label", "bounds", "confidence", "current_mode", "target_mode"}
        ):
            raw_switch_current_mode = raw_mode_switch.get("current_mode")
            discardable_mode_switch_semantic_conflict = bool(
                raw_switch_current_mode
                in {"direct_latin", "chinese_pinyin"}
                and keyboard_input_mode in {"direct_latin", "chinese_pinyin"}
                and raw_switch_current_mode != keyboard_input_mode
            )
        if (
            not switch_is_goal
            and not input_needs_mode_switch
            and trusted_input is not None
            and keyboard_visible
            and keyboard_layout == "qwerty"
            and keyboard_input_mode in {"direct_latin", "chinese_pinyin"}
            and (
                _is_incomplete_optional_keyboard_mode_switch(raw_mode_switch)
                or discardable_mode_switch_geometry
                or discardable_mode_switch_semantic_conflict
            )
        ):
            # The current deterministic segment does not consume this optional
            # control. Revoke a missing-field subset or an exact-shaped claim
            # whose coordinates or semantics conflict with the independently
            # audited whole-keyboard mode. When switching is the actual next
            # action, the same contradictions remain strict.
            mode_switch = None
        else:
            mode_switch = _validated_keyboard_mode_switch(
                raw_mode_switch,
                keyboard_bounds=keyboard_bounds,
            )
        if (
            mode_switch is not None
            and keyboard_input_mode != "unknown"
            and mode_switch["current_mode"] != keyboard_input_mode
        ):
            raise UISceneError("模式切换键 current_mode 与键盘 input_mode 冲突。")
        if input_needs_mode_switch and (
            mode_switch is None
            or mode_switch["target_mode"] != input_step.required_mode
        ):
            raise UISceneError("模式切换键未绑定下一确定性文字分段所需方向。")
        literal_keys = _validated_keyboard_literal_keys(
            keyboard.get("literal_keys", []),
            keyboard_bounds=keyboard_bounds,
        )
        literal_key_targets = set(
            _input_audit_literal_key_targets(
                _observation_goal_context(goal_context),
                current_input_text=(
                    trusted_input["text"]
                    if trusted_input is not None
                    else None
                ),
            )
        )
        # Schema-valid visible keys outside the locally derived unique next
        # character are non-authoritative observations.  Project them away;
        # never let them become scene elements or action candidates.
        literal_keys = [
            item for item in literal_keys if item["value"] in literal_key_targets
        ]
        if keyboard_layout == "qwerty" and qwerty_geometry is not None:
            literal_keys = [
                item
                for item in literal_keys
                if not _literal_key_overlaps_qwerty_letter_cell(
                    item,
                    qwerty_geometry=qwerty_geometry,
                )
            ]
        layout_switches = _validated_keyboard_layout_switches(
            keyboard.get("layout_switches", []),
            keyboard_bounds=keyboard_bounds,
            current_layout=keyboard_layout,
        )
        input_needs_case_switch = bool(
            input_step is not None
            and input_step.kind == "direct_latin"
            and bool(input_step.required_case_mode)
            and keyboard_case_mode != input_step.required_case_mode
        )
        raw_case_switch = keyboard.get("case_switch")
        if (
            not input_needs_case_switch
            and isinstance(raw_case_switch, dict)
            and set(raw_case_switch)
            == {"label", "bounds", "confidence", "current_mode", "target_mode"}
            and not _valid_1000_bounds(raw_case_switch.get("bounds"))
        ):
            raw_case_switch = None
        case_switch = _validated_keyboard_case_switch(
            raw_case_switch,
            keyboard_bounds=keyboard_bounds,
            keyboard_layout=keyboard_layout,
            keyboard_input_mode=keyboard_input_mode,
            case_mode=keyboard_case_mode,
        )
        exact_literal_key: dict[str, Any] | None = None
        exact_enter_key: dict[str, Any] | None = None
        exact_layout_switch: dict[str, Any] | None = None
        exact_case_switch: dict[str, Any] | None = None
        if input_step is not None:
            if (
                input_step.kind in {"direct_latin", "chinese_pinyin"}
                and keyboard_layout != "qwerty"
            ):
                exact_switches = [
                    item for item in layout_switches
                    if item["target_layout"] == "qwerty"
                ]
                if len(exact_switches) == 1:
                    exact_layout_switch = exact_switches[0]
            elif (
                input_step.kind == "literal_key"
                and input_step.segment == "\n"
            ):
                if (
                    active_multiline
                    and enter_key is not None
                    and enter_key["key_action"] == "newline"
                ):
                    exact_enter_key = enter_key
            elif input_step.kind == "literal_key":
                exact_keys = [
                    item for item in literal_keys
                    if item["value"] == input_step.segment
                ]
                if len(exact_keys) == 1:
                    exact_literal_key = exact_keys[0]
                else:
                    desired_layout = _preferred_keyboard_layout(
                        input_step.segment
                    )
                    exact_switches = [
                        item for item in layout_switches
                        if item["target_layout"] == desired_layout
                    ]
                    if (
                        desired_layout != keyboard_layout
                        and len(exact_switches) == 1
                    ):
                        exact_layout_switch = exact_switches[0]
            elif (
                input_step.kind == "direct_latin"
                and bool(input_step.required_case_mode)
                and keyboard_case_mode != input_step.required_case_mode
                and case_switch is not None
                and case_switch["target_mode"] == input_step.required_case_mode
            ):
                exact_case_switch = case_switch
        if (
            trusted_input is not None
            and keyboard_visible
            and keyboard_layout == "qwerty"
            and keyboard_input_mode in {"direct_latin", "chinese_pinyin"}
            and _goal_has_explicit_input_text(goal_context)
            and not switch_is_goal
            and input_step is not None
            and input_step.kind in {"direct_latin", "chinese_pinyin"}
            and exact_case_switch is None
            and qwerty_geometry is None
            and not typed_prefix_verification_only
            and not active_clear_goal
        ):
            raise UISceneError(
                "文字输入授权要求本轮输入结构审计提供有效 QWERTY anchors。"
            )
        rendered_input = trusted_input or (
            predecessor_input if next_field_key is not None else None
        )

        value = scene.to_dict()
        elements: list[dict[str, Any]] = []
        for element in value.get("elements") or []:
            if not isinstance(element, dict):
                continue
            element = dict(element)
            element["states"] = dict(element.get("states") or {})
            element["states"]["goal_relevant"] = False
            # Compact input proposals never survive the dedicated audit.  The
            # typed ledger is the only published input state even when this
            # audit cannot establish a replacement, so an old compact value
            # cannot reappear through the early-return path.
            if (
                element.get("role") == "input"
                or element.get("meaning") == "application_text_input"
            ):
                continue
            elements.append(element)
        if (
            trusted_input is None
            and next_field_key is None
            and (mode_switch is None or not switch_is_goal)
        ):
            value["elements"] = elements
            value["summary"] = (
                "typed输入状态账本未建立；compact输入摘要与输入转写不参与判断。"
            )
            return UIScene.from_dict(
                value,
                coordinate_scale=1.0,
                stable_override=True,
                fingerprint_override=fingerprint,
            )
        if rendered_input is not None:
            pending_auxiliary_input_action = any(
                item is not None
                for item in (
                    exact_ime_candidate,
                    exact_literal_key,
                    exact_enter_key,
                    exact_layout_switch,
                    exact_case_switch,
                )
            )
            input_requires_auxiliary_action = bool(
                not active_clear_goal
                and input_step is not None
                and keyboard_visible
                and (
                    input_step.kind == "literal_key"
                    or (
                        input_step.kind in {"direct_latin", "chinese_pinyin"}
                        and keyboard_layout != "qwerty"
                    )
                    or (
                        bool(input_step.required_case_mode)
                        and keyboard_case_mode != input_step.required_case_mode
                    )
                )
            )
            states: dict[str, Any] = {
                "goal_relevant": (
                    not switch_is_goal
                    and next_field_key is None
                    and not pending_auxiliary_input_action
                    and not input_requires_auxiliary_action
                    and not typed_prefix_verification_only
                ),
                "fully_visible": True,
                "value": rendered_input["text"],
            }
            rendered_field_id = (
                predecessor_field_id if next_field_key is not None else active_field_id
            )
            rendered_field_label = (
                predecessor_field_label
                if next_field_key is not None
                else active_field_label
            )
            if rendered_field_id:
                states["input_field_id"] = rendered_field_id
                if next_field_key is None:
                    states["input_multiline"] = active_multiline
            if rendered_field_label:
                states["input_field_label"] = rendered_field_label
            if rendered_input.get("verified_trailing_newline") is True:
                states["verified_trailing_newline"] = True
            if isinstance(rendered_input.get("local_caret_line_index"), int):
                states["local_caret_line_index"] = rendered_input[
                    "local_caret_line_index"
                ]
            clear_extra_delete_units = rendered_input.get(
                "clear_extra_delete_units"
            )
            if (
                isinstance(clear_extra_delete_units, int)
                and not isinstance(clear_extra_delete_units, bool)
                and clear_extra_delete_units > 0
            ):
                states["clear_extra_delete_units"] = clear_extra_delete_units
            if not keyboard_visible:
                # Absence is useful task evidence only when the dedicated
                # full-frame input audit explicitly reports keyboard.visible=false.
                # An empty preliminary overlays list alone never mints this fact.
                states["soft_keyboard_visible"] = False
            if rendered_input["placeholder"]:
                states["placeholder"] = rendered_input["placeholder"]
            if keyboard_bounds is not None or boundsless_keyboard_dismissal:
                states.update(
                    {
                        "focused": True,
                        "keyboard_layout": keyboard_layout,
                        "keyboard_input_mode": keyboard_input_mode,
                        "keyboard_case_mode": keyboard_case_mode,
                    }
                )
            if qwerty_geometry is not None:
                states["keyboard_geometry"] = qwerty_geometry
            elif generic_backspace_geometry is not None:
                states["keyboard_geometry"] = {
                    "type": "generic",
                    "anchors": {
                        "backspace": generic_backspace_geometry["center"]
                    },
                    "source": "input_structure_audit",
                }
            if exact_ime_candidate is not None and input_step is not None:
                states.update(
                    {
                        "ime_preedit_text": exact_ime_preedit_text,
                        "ime_exact_candidate_text": input_step.segment,
                    }
                )
            elif clearable_ime_preedit:
                states["ime_preedit_text"] = clearable_ime_preedit
            input_label = rendered_input["text"] or rendered_input["placeholder"]
            # ``field_labels`` are literal, field-attached observations from
            # the dedicated input audit.  Preserve them as exact candidate
            # evidence so a labelled input remains addressable after the
            # preliminary scene element is replaced by this audited input.
            input_evidence = list(
                dict.fromkeys(
                    (
                        *rendered_input["field_labels"],
                        *rendered_input["visible_editable_cues"],
                    )
                )
            )
            if rendered_input["text"]:
                input_evidence.insert(0, f"应用输入框当前文字：{rendered_input['text']}")
                lineage_visual_text = rendered_input.get("lineage_visual_text")
                if isinstance(lineage_visual_text, str):
                    input_evidence.append(
                        f"视觉折行转写：{lineage_visual_text}；本地逐键连续性逐字核对通过"
                    )
                lineage_visible_cue_text = rendered_input.get(
                    "lineage_visible_cue_text"
                )
                if isinstance(lineage_visible_cue_text, str):
                    input_evidence.append(
                        "输入状态切换后同一应用输入区域仍逐字可见："
                        f"{lineage_visible_cue_text}；本地同值连续性核对通过"
                    )
                lineage_persisted_visible_cue_text = rendered_input.get(
                    "lineage_persisted_visible_cue_text"
                )
                if isinstance(lineage_persisted_visible_cue_text, str):
                    input_evidence.append(
                        "跨会话同一应用输入区域仍逐字可见："
                        f"{lineage_persisted_visible_cue_text}；持久回执连续性核对通过"
                    )
                lineage_pending_text_prefix = rendered_input.get(
                    "lineage_pending_text_prefix"
                )
                if isinstance(lineage_pending_text_prefix, str):
                    input_evidence.append(
                        "pending typed连续性、授权payload与唯一预编辑后缀"
                        "共同确认已提交前缀："
                        f"{lineage_pending_text_prefix}"
                    )
                lineage_ime_candidate_committed_value = rendered_input.get(
                    "lineage_ime_candidate_committed_value"
                )
                if isinstance(lineage_ime_candidate_committed_value, str):
                    input_evidence.append(
                        "候选点击回执、typed字段、授权payload与动作后同字段精确片段"
                        "共同确认已提交中文："
                        f"{lineage_ime_candidate_committed_value}；残留预测栏未作为预编辑"
                    )
                authorized_prefix_visible_cue_text = rendered_input.get(
                    "authorized_prefix_visible_cue_text"
                )
                if isinstance(authorized_prefix_visible_cue_text, str):
                    input_evidence.append(
                        "typed字段内唯一可见文字逐字匹配授权payload前缀："
                        f"{authorized_prefix_visible_cue_text}；同帧无IME预编辑"
                    )
                if rendered_input.get("verified_trailing_newline") is True:
                    input_evidence.append(
                        "已验证换行动作、同一typed输入框、精确可见前缀与下一行光标一致"
                    )
                if isinstance(rendered_input.get("local_caret_line_index"), int):
                    input_evidence.append(
                        "模型确认可见光标后，本地校准像素定位光标视觉行："
                        f"{rendered_input['local_caret_line_index']}"
                    )
                clear_goal_visible_cue_text = rendered_input.get(
                    "clear_goal_visible_cue_text"
                )
                if isinstance(clear_goal_visible_cue_text, str):
                    input_evidence.append(
                        "清空目标的同帧唯一已提交可见文字："
                        f"{clear_goal_visible_cue_text}；额外视觉行仅作为保守退格单位"
                    )
            elif rendered_input["placeholder"]:
                input_evidence.insert(0, f"应用输入框为空，占位提示：{rendered_input['placeholder']}")
            if clearable_ime_preedit:
                input_evidence.append(
                    "唯一相邻输入法预编辑串已绑定当前typed输入框，"
                    f"可清理文字：{clearable_ime_preedit}"
                )
            if not keyboard_visible:
                input_evidence.append(AUDITED_SOFT_KEYBOARD_HIDDEN_EVIDENCE)
            elements.append(
                {
                    "element_id": "local_audited_input_1",
                    "role": "input",
                    "meaning": "application_text_input",
                    "label": input_label,
                    "bounds": [part / 1000.0 for part in rendered_input["input_bounds"]],
                    "confidence": rendered_input["confidence"],
                    "states": states,
                    "evidence": input_evidence[:6],
                }
            )
            right_button = rendered_input["right_button"]
            if right_button is not None:
                elements.append(
                    {
                        "element_id": "local_audited_adjacent_button_1",
                        "role": "button",
                        "meaning": "adjacent_input_utility",
                        "label": right_button["label"],
                        "bounds": [part / 1000.0 for part in right_button["bounds"]],
                        "confidence": right_button["confidence"],
                        "states": {"goal_relevant": False, "fully_visible": True},
                        "evidence": ["应用输入结构的相邻独立控件；不具备目标权限"],
                    }
                )
        if next_field_key is not None:
            elements.append(_next_field_key_element(
                next_field_key,
                source_field_id=predecessor_field_id,
                target_field_id=active_field_id,
                target_field_label=active_field_label,
            ))
        if exact_ime_candidate is not None and input_step is not None:
            elements.append(
                {
                    "element_id": "local_audited_ime_candidate_1",
                    "role": "button",
                    "meaning": "ime_exact_candidate",
                    "label": exact_ime_candidate["text"],
                    "bounds": [part / 1000.0 for part in exact_ime_candidate["bounds"]],
                    "confidence": exact_ime_candidate["confidence"],
                    "states": {
                        "goal_relevant": True,
                        "fully_visible": True,
                        "ime_candidate": True,
                        "input_element_id": "local_audited_input_1",
                        "prior_input_value": input_step.current_text,
                        "expected_input_value": input_step.expected_value,
                        "pinyin": exact_ime_preedit_text,
                    },
                    "evidence": [
                        "输入结构审计确认当前输入法组合的唯一逐字候选："
                        f"{input_step.segment}"
                    ],
                }
            )
        if exact_literal_key is not None and input_step is not None:
            elements.append(
                {
                    "element_id": "local_audited_literal_key_1",
                    "role": "button",
                    "meaning": "input_exact_literal_key",
                    "label": exact_literal_key["label"],
                    "bounds": [part / 1000.0 for part in exact_literal_key["bounds"]],
                    "confidence": exact_literal_key["confidence"],
                    "states": {
                        "goal_relevant": True,
                        "fully_visible": True,
                        "input_literal_key": True,
                        "key_value": input_step.segment,
                        "prior_input_value": input_step.current_text,
                        "expected_input_value": input_step.expected_value,
                        "input_element_id": "local_audited_input_1",
                    },
                    "evidence": [
                        "输入结构审计确认下一字符对应唯一完整可见键位"
                    ],
                }
            )
        if exact_enter_key is not None and input_step is not None:
            elements.append(
                {
                    "element_id": "local_audited_enter_key_1",
                    "role": "button",
                    "meaning": "input_exact_enter_key",
                    "label": exact_enter_key["label"],
                    "bounds": [
                        part / 1000.0 for part in exact_enter_key["bounds"]
                    ],
                    "confidence": exact_enter_key["confidence"],
                    "states": {
                        "goal_relevant": True,
                        "fully_visible": True,
                        "input_enter_key": True,
                        "key_action": "newline",
                        "key_value": "\n",
                        "prior_input_value": input_step.current_text,
                        "expected_input_value": input_step.expected_value,
                        "input_element_id": "local_audited_input_1",
                        "input_field_id": active_field_id,
                    },
                    "evidence": [
                        "输入结构审计确认当前多行字段的唯一完整可见换行键"
                    ],
                }
            )
        if exact_layout_switch is not None and input_step is not None:
            elements.append(
                {
                    "element_id": "local_audited_keyboard_layout_switch_1",
                    "role": "button",
                    "meaning": "switch_keyboard_layout",
                    "label": exact_layout_switch["label"],
                    "bounds": [part / 1000.0 for part in exact_layout_switch["bounds"]],
                    "confidence": exact_layout_switch["confidence"],
                    "states": {
                        "goal_relevant": True,
                        "fully_visible": True,
                        "keyboard_layout_switch": True,
                        "current_layout": exact_layout_switch["current_layout"],
                        "target_layout": exact_layout_switch["target_layout"],
                        "prior_input_value": input_step.current_text,
                        "next_input_value": input_step.segment,
                        "input_element_id": "local_audited_input_1",
                    },
                    "evidence": ["输入结构审计确认方向明确的键盘布局切换键"],
                }
            )
        if exact_case_switch is not None and input_step is not None:
            elements.append(
                {
                    "element_id": "local_audited_keyboard_case_switch_1",
                    "role": "button",
                    "meaning": "switch_keyboard_case",
                    "label": exact_case_switch["label"],
                    "bounds": [part / 1000.0 for part in exact_case_switch["bounds"]],
                    "confidence": exact_case_switch["confidence"],
                    "states": {
                        "goal_relevant": True,
                        "fully_visible": True,
                        "keyboard_case_switch": True,
                        "current_mode": exact_case_switch["current_mode"],
                        "target_mode": exact_case_switch["target_mode"],
                        "prior_input_value": input_step.current_text,
                        "input_element_id": "local_audited_input_1",
                    },
                    "evidence": ["输入结构审计确认方向明确的大小写切换键"],
                }
            )
        if mode_switch is not None:
            elements.append(
                {
                    "element_id": "local_audited_keyboard_mode_switch_1",
                    "role": "button",
                    "meaning": "switch_keyboard_input_mode",
                    "label": mode_switch["label"],
                    "bounds": [part / 1000.0 for part in mode_switch["bounds"]],
                    "confidence": mode_switch["confidence"],
                    "states": {
                        "goal_relevant": switch_is_goal,
                        "fully_visible": True,
                        "keyboard_input_mode_switch": True,
                        "current_mode": mode_switch["current_mode"],
                        "target_mode": mode_switch["target_mode"],
                        "prior_input_value": (
                            input_step.current_text if input_step is not None else ""
                        ),
                        "next_input_value": (
                            input_step.segment if input_step is not None else ""
                        ),
                        "input_element_id": "local_audited_input_1",
                    },
                    "evidence": ["键盘区域内方向明确的独立输入模式切换键"],
                }
            )
        for element in elements:
            if not str(element.get("element_id") or "").startswith(
                "local_audited_"
            ):
                continue
            states = element.get("states")
            if (
                not isinstance(states, dict)
                or states.get("fully_visible") is not True
                or not element.get("evidence")
            ):
                continue
            states["primary_input_geometry_verified"] = True
            states["geometry_audit_source"] = "input_structure_audit"
        value["elements"] = elements
        value["summary"] = (
            "输入状态仅见typed输入账本；compact输入摘要与输入转写不参与判断。"
        )
        return UIScene.from_dict(
            value,
            coordinate_scale=1.0,
            stable_override=True,
            fingerprint_override=fingerprint,
        )
    except (UISceneError, ValueError, TypeError) as exc:
        raise VisionAgentError(f"输入结构只读审计结果不符合协议：{exc}") from exc


def _apply_hidden_keyboard_only_attestation(
    scene: UIScene,
    raw: str,
    *,
    fingerprint: str,
    goal_context: dict[str, Any],
) -> UIScene:
    """Keep only a strict hidden-keyboard fact when input geometry is invalid.

    This path is completion-only.  It never consumes application input bounds
    or adjacent-control geometry.  It accepts only one fully visible input whose
    exact text equals the controller-owned canonical input text, then stores the
    value and hidden-keyboard facts as non-geometric summary evidence.  Every
    preliminary element loses goal relevance, so no action can be authorized
    from model-estimated geometry.
    """

    try:
        if not _goal_requests_keyboard_dismissal(goal_context):
            raise UISceneError("当前子目标不是软键盘收起验证。")
        payload = _extract_json_object(raw)
        if set(payload) != {
            "protocol_version",
            "application_inputs",
            "ime_preedit_regions",
            "keyboard",
        }:
            raise UISceneError("输入结构审计包含协议外字段。")
        if payload.get("protocol_version") != INPUT_STRUCTURE_AUDIT_VERSION:
            raise UISceneError("输入结构审计协议版本不匹配。")
        application_inputs = payload.get("application_inputs")
        ime_preedit_regions = payload.get("ime_preedit_regions")
        if (
            not isinstance(application_inputs, list)
            or len(application_inputs) > 4
            or not isinstance(ime_preedit_regions, list)
            or len(ime_preedit_regions) > 4
        ):
            raise UISceneError("输入结构审计数组字段无效。")
        keyboard = payload.get("keyboard")
        hidden_required = {"visible", "bounds", "layout", "input_mode", "mode_switch"}
        hidden_optional = {
            "qwerty_anchors", "backspace_key", "case_mode", "case_switch",
            "literal_keys", "layout_switches",
        }
        if (
            not isinstance(keyboard, dict)
            or not hidden_required.issubset(keyboard)
            or set(keyboard) - hidden_required - hidden_optional
        ):
            raise UISceneError("输入结构审计 keyboard 字段不符合协议。")
        if not (
            keyboard.get("visible") is False
            and keyboard.get("bounds") is None
            and keyboard.get("layout") == "unknown"
            and keyboard.get("input_mode") == "unknown"
            and keyboard.get("mode_switch") is None
            and keyboard.get("qwerty_anchors") is None
            and keyboard.get("backspace_key") is None
            and keyboard.get("case_mode", "unknown") == "unknown"
            and keyboard.get("case_switch") is None
            and keyboard.get("literal_keys") in (None, [])
            and keyboard.get("layout_switches") in (None, [])
        ):
            raise UISceneError("软键盘不可见事实不完整。")
        focused = _active_subgoal_visual_context(goal_context)
        entities = (
            focused.get("goal_entities")
            if focused is not goal_context
            else goal_context.get("entities")
        )
        canonical_text = (
            str(entities.get("input_text") or "").strip()
            if isinstance(entities, dict)
            else ""
        )
        if not canonical_text or len(application_inputs) != 1 or ime_preedit_regions:
            raise UISceneError("软键盘收起验证缺少唯一 canonical 输入值。")
        input_item = application_inputs[0]
        if not isinstance(input_item, dict) or set(input_item) != {
            "structure_id",
            "bounds",
            "fully_visible",
            "text",
            "placeholder",
            "visible_editable_cues",
            "confidence",
            "right_button",
        } and set(input_item) != {
            "structure_id",
            "bounds",
            "fully_visible",
            "text",
            "placeholder",
            "visible_editable_cues",
            "caret_line_index",
            "confidence",
            "right_button",
        }:
            raise UISceneError("应用输入结构字段不符合协议。")
        rejected_bounds = input_item.get("bounds")
        cues = input_item.get("visible_editable_cues")
        right_button = input_item.get("right_button")
        if (
            not str(input_item.get("structure_id") or "").strip()
            or not isinstance(rejected_bounds, list)
            or len(rejected_bounds) != 4
            or any(
                isinstance(part, bool) or not isinstance(part, (int, float))
                for part in rejected_bounds
            )
            or input_item.get("fully_visible") is not True
            or input_item.get("text") != canonical_text
            or not isinstance(input_item.get("placeholder"), str)
            or not isinstance(cues, list)
            or not cues
            or len(cues) > 4
            or any(not isinstance(part, str) or not part.strip() for part in cues)
            or input_item.get("caret_line_index") is not None
            or _audit_confidence(input_item.get("confidence"), "应用输入结构") < 0.9
            or (
                right_button is not None
                and (
                    not isinstance(right_button, dict)
                    or not set(right_button) <= {"label", "bounds", "confidence"}
                )
            )
        ):
            raise UISceneError("应用输入值只读事实不完整。")
        keyboard_pattern = re.compile(
            r"(?:软键盘|输入法|键盘|keyboard|ime)",
            re.IGNORECASE,
        )
        if any(keyboard_pattern.search(item) for item in scene.overlays) or any(
            keyboard_pattern.search(
                " ".join((item.role, item.meaning, item.label, *item.evidence))
            )
            for item in scene.elements
        ):
            raise UISceneError("基础场景仍包含结构化可见键盘，证据冲突。")
        value = scene.to_dict()
        elements: list[dict[str, Any]] = []
        for item in value.get("elements") or []:
            item = dict(item)
            states = dict(item.get("states") or {})
            states["goal_relevant"] = False
            item["states"] = states
            elements.append(item)
        value["elements"] = elements
        summary_facts = (
            f"输入结构只读审计确认应用输入框当前文字：{canonical_text}",
            AUDITED_SOFT_KEYBOARD_HIDDEN_EVIDENCE,
        )
        summary = str(value.get("summary") or "").strip()
        for fact in summary_facts:
            if fact not in summary:
                summary = f"{summary}；{fact}" if summary else fact
        value["summary"] = summary
        return UIScene.from_dict(
            value,
            coordinate_scale=1.0,
            stable_override=True,
            fingerprint_override=fingerprint,
        )
    except (UISceneError, ValueError, TypeError) as exc:
        raise VisionAgentError(
            f"软键盘收起只读事实不符合隔离合同：{exc}"
        ) from exc


def _audit_confidence(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UISceneError(f"{field_name} confidence 格式无效。")
    confidence = float(value)
    if not 0.0 <= confidence <= 1.0:
        raise UISceneError(f"{field_name} confidence 超出0..1。")
    return confidence


def _keyboard_bounds_from_locally_snapped_qwerty_anchors(
    value: Any,
) -> tuple[float, float, float, float] | None:
    """Rebuild one keyboard envelope from independently stabilized QWERTY rows.

    The model's outer rectangle is not a second authority once all seven
    anchors have been stabilized across frames. Individual Enter, literal,
    switch and backspace controls still require their own audited geometry.
    """

    expected = {"q", "p", "a", "l", "z", "m", "backspace"}
    if not isinstance(value, dict) or set(value) != expected:
        return None
    try:
        anchors = {
            key: [float(value[key][0]), float(value[key][1])]
            for key in expected
            if isinstance(value.get(key), (list, tuple))
            and len(value[key]) == 2
            and all(
                not isinstance(part, bool) and isinstance(part, (int, float))
                for part in value[key]
            )
        }
        if set(anchors) != expected or any(
            not 0 <= coordinate <= 1000
            for point in anchors.values()
            for coordinate in point
        ):
            return None
        normalized = {
            key: [round(point[0]), round(point[1])]
            for key, point in anchors.items()
        }
        qwerty_keyboard_config_from_anchors(normalized)
        horizontal_pitch = min(
            (anchors["p"][0] - anchors["q"][0]) / 9.0,
            (anchors["l"][0] - anchors["a"][0]) / 8.0,
            (anchors["m"][0] - anchors["z"][0]) / 6.0,
        )
        top_y = (anchors["q"][1] + anchors["p"][1]) / 2.0
        middle_y = (anchors["a"][1] + anchors["l"][1]) / 2.0
        bottom_y = (
            anchors["z"][1]
            + anchors["m"][1]
            + anchors["backspace"][1]
        ) / 3.0
        vertical_pitch = min(middle_y - top_y, bottom_y - middle_y)
        if horizontal_pitch <= 0 or vertical_pitch <= 0:
            return None
        bounds = (
            max(
                0.0,
                min(anchors[key][0] for key in ("q", "a", "z"))
                - 0.75 * horizontal_pitch,
            ),
            max(0.0, top_y - 0.75 * vertical_pitch),
            min(
                1000.0,
                max(
                    anchors[key][0]
                    for key in ("p", "l", "m", "backspace")
                )
                + 0.75 * horizontal_pitch,
            ),
            min(1000.0, bottom_y + 2.0 * vertical_pitch),
        )
        if bounds[2] - bounds[0] < 300 or bounds[3] - bounds[1] < 180:
            return None
        return bounds
    except (KeyError, TypeError, ValueError, WorkflowNotReady):
        return None


def _validated_qwerty_keyboard_geometry(
    value: Any,
    *,
    keyboard_bounds: tuple[float, float, float, float],
    locally_snapped: bool = False,
) -> dict[str, Any]:
    """Mint a locally checked execution profile from current-frame facts."""

    if not isinstance(value, dict):
        raise UISceneError("QWERTY anchors 必须是对象。")
    expected = {"q", "p", "a", "l", "z", "m", "backspace"}
    if set(value) != expected:
        raise UISceneError("QWERTY anchors 必须精确包含 q/p/a/l/z/m/backspace。")
    normalized: dict[str, list[int]] = {}
    for key in sorted(expected):
        point = value.get(key)
        if (
            not isinstance(point, (list, tuple))
            or len(point) != 2
            or any(
                isinstance(part, bool) or not isinstance(part, (int, float))
                for part in point
            )
        ):
            raise UISceneError(f"QWERTY anchor {key} 格式无效。")
        x, y = float(point[0]), float(point[1])
        within_horizontal_bounds = (
            keyboard_bounds[0] - 20 <= x <= keyboard_bounds[2] + 20
        )
        within_vertical_bounds = (
            keyboard_bounds[1] - 20 <= y <= keyboard_bounds[3] + 20
        )
        if not within_horizontal_bounds or (
            not locally_snapped and not within_vertical_bounds
        ):
            raise UISceneError(f"QWERTY anchor {key} 不在已审计键盘区域内。")
        normalized[key] = [round(x), round(y)]
    try:
        qwerty_keyboard_config_from_anchors(normalized)
    except WorkflowNotReady as exc:
        raise UISceneError(f"QWERTY anchors 未通过本地布局校验：{exc}") from exc
    return {
        "type": "qwerty",
        "anchors": normalized,
        "source": "input_structure_audit",
    }


def _validated_keyboard_mode_switch(
    value: Any,
    *,
    keyboard_bounds: tuple[float, float, float, float] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if keyboard_bounds is None:
        raise UISceneError("模式切换键必须绑定可见键盘区域。")
    if not isinstance(value, dict) or set(value) != {
        "label",
        "bounds",
        "confidence",
        "current_mode",
        "target_mode",
    }:
        raise UISceneError("输入结构审计 mode_switch 字段不符合协议。")
    if not _valid_1000_bounds(value.get("bounds")):
        raise UISceneError("输入结构审计 mode_switch bounds 无效。")
    confidence = _audit_confidence(value.get("confidence"), "mode_switch")
    current_mode = value.get("current_mode")
    target_mode = value.get("target_mode")
    modes = {"direct_latin", "chinese_pinyin"}
    if current_mode not in modes or target_mode not in modes or current_mode == target_mode:
        raise UISceneError("mode_switch 必须给出方向明确且不同的输入模式。")
    label = str(value.get("label") or "").strip()
    bounds = tuple(float(part) for part in value["bounds"])
    keyboard_width = keyboard_bounds[2] - keyboard_bounds[0]
    keyboard_height = keyboard_bounds[3] - keyboard_bounds[1]
    if (
        confidence < 0.9
        or not label
        or not _is_explicit_keyboard_mode_label(label)
        or not _bounds_inside(bounds, keyboard_bounds, tolerance=20)
        or bounds[2] - bounds[0] > 0.35 * keyboard_width
        or bounds[3] - bounds[1] > 0.30 * keyboard_height
    ):
        return None
    return {
        "label": label,
        "bounds": [round(part) for part in bounds],
        "confidence": confidence,
        "current_mode": current_mode,
        "target_mode": target_mode,
    }


def _validated_keyboard_backspace_key(
    value: Any,
    *,
    keyboard_bounds: tuple[float, float, float, float] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if keyboard_bounds is None:
        raise UISceneError("退格键必须绑定完整可见键盘区域。")
    if not isinstance(value, dict) or set(value) != {
        "label",
        "bounds",
        "confidence",
        "fully_visible",
    }:
        raise UISceneError("输入结构审计 backspace_key 字段不符合协议。")
    label = str(value.get("label") or "").strip()
    if not label or not re.search(
        r"(?:⌫|⌦|退格|删除|backspace|delete)",
        label,
        re.IGNORECASE,
    ):
        # An exact-shaped but semantically unsupported optional claim cannot
        # authorize deletion. Revoke only this candidate so independent input
        # facts or a literal key can still be used; a clear operation will
        # remain blocked because no backspace geometry is minted.
        return None
    if value.get("fully_visible") is not True or not _valid_1000_bounds(
        value.get("bounds")
    ):
        return None
    confidence = _audit_confidence(value.get("confidence"), "backspace_key")
    bounds = tuple(float(part) for part in value["bounds"])
    if confidence < 0.9 or not _bounds_inside(
        bounds,
        keyboard_bounds,
        tolerance=12,
    ):
        return None
    return {
        "label": label,
        "center": [
            round((bounds[0] + bounds[2]) / 2),
            round((bounds[1] + bounds[3]) / 2),
        ],
        "confidence": confidence,
    }


def _validated_keyboard_enter_key(
    value: Any,
    *,
    keyboard_bounds: tuple[float, float, float, float] | None,
) -> dict[str, Any] | None:
    """Validate a visible keyboard action key without changing its semantics."""

    if value is None:
        return None
    if keyboard_bounds is None:
        raise UISceneError("回车键必须绑定完整可见键盘区域。")
    if not isinstance(value, dict) or set(value) != {
        "label",
        "bounds",
        "confidence",
        "fully_visible",
        "key_action",
    }:
        raise UISceneError("输入结构审计 enter_key 字段不符合协议。")
    key_action = value.get("key_action")
    if key_action not in {"newline", "send", "search", "done", "next", "unknown"}:
        raise UISceneError("输入结构审计 enter_key.key_action 无效。")
    label = str(value.get("label") or "").strip()
    if (
        not label
        or value.get("fully_visible") is not True
        or not _valid_1000_bounds(value.get("bounds"))
    ):
        return None
    confidence = _audit_confidence(value.get("confidence"), "enter_key")
    bounds = tuple(float(part) for part in value["bounds"])
    if confidence < 0.9 or not _bounds_inside(
        bounds,
        keyboard_bounds,
        tolerance=12,
    ):
        return None
    return {
        "label": label,
        "bounds": [round(part) for part in bounds],
        "confidence": confidence,
        "key_action": key_action,
    }


def _locally_snapped_keyboard_enter_key(
    value: Any,
    *,
    anchors: dict[str, list[int]],
    keyboard_bounds: tuple[float, float, float, float] | None,
) -> dict[str, Any] | None:
    """Bind an audited Enter/Next semantic to stable local QWERTY geometry.

    Qwen remains responsible for identifying the visible key and its current
    newline semantic.  Its rectangle is not execution authority once local OCR
    has independently stabilized the whole QWERTY grid; only the horizontal
    vicinity of the reported key is retained as a cross-check.  An icon-only
    newline key may omit its text label, but only after that local geometry and
    Qwen's explicit ``key_action=newline`` agree; conflicting labels and
    unlabeled Next keys remain rejected.
    """

    if keyboard_bounds is None or not isinstance(value, dict) or set(value) != {
        "label",
        "bounds",
        "confidence",
        "fully_visible",
        "key_action",
    }:
        return None
    key_action = value.get("key_action")
    if value.get("fully_visible") is not True or key_action not in {
        "newline",
        "next",
    }:
        return None
    label = str(value.get("label") or "").strip()
    normalized_label = re.sub(r"\s+", "", label).casefold()
    newline_label = bool(
        any(glyph in label for glyph in ("↵", "⏎", "⤶", "⮐"))
        or normalized_label in {"enter", "return", "回车", "换行"}
    )
    next_label = bool(
        any(glyph in label for glyph in ("→", "↦", "➡", "⏭"))
        or normalized_label in {"next", "下一步", "下一个", "下一项"}
    )
    unlabeled_newline = key_action == "newline" and not normalized_label
    if not (
        (key_action == "newline" and (newline_label or unlabeled_newline))
        or (key_action == "next" and next_label)
    ):
        return None
    confidence = _audit_confidence(value.get("confidence"), "enter_key")
    raw_bounds = value.get("bounds")
    if (
        confidence < 0.9
        or not isinstance(raw_bounds, (list, tuple))
        or len(raw_bounds) != 4
        or any(
            isinstance(part, bool) or not isinstance(part, (int, float))
            for part in raw_bounds
        )
    ):
        return None
    raw_left, raw_top, raw_right, raw_bottom = (
        float(part) for part in raw_bounds
    )
    if not (
        0 <= raw_left < raw_right <= 1000
        and math.isfinite(raw_top)
        and math.isfinite(raw_bottom)
        and raw_top < raw_bottom
    ):
        return None
    try:
        q_x = float(anchors["q"][0])
        p_x = float(anchors["p"][0])
        backspace_x = float(anchors["backspace"][0])
        top_y = statistics.mean(
            (float(anchors["q"][1]), float(anchors["p"][1]))
        )
        middle_y = statistics.mean(
            (float(anchors["a"][1]), float(anchors["l"][1]))
        )
        bottom_y = statistics.mean(
            (
                float(anchors["z"][1]),
                float(anchors["m"][1]),
                float(anchors["backspace"][1]),
            )
        )
    except (KeyError, TypeError, ValueError):
        return None
    horizontal_pitch = (p_x - q_x) / 9.0
    vertical_pitch = statistics.mean(
        (middle_y - top_y, bottom_y - middle_y)
    )
    if not (45 <= horizontal_pitch <= 130 and 35 <= vertical_pitch <= 140):
        return None
    raw_center_x = (raw_left + raw_right) / 2.0
    if abs(raw_center_x - backspace_x) > max(2.0 * horizontal_pitch, 180.0):
        return None
    center_y = bottom_y + vertical_pitch
    half_width = 0.65 * horizontal_pitch
    half_height = 0.45 * vertical_pitch
    snapped_bounds = (
        max(keyboard_bounds[0], backspace_x - half_width),
        max(keyboard_bounds[1], center_y - half_height),
        min(keyboard_bounds[2], backspace_x + half_width),
        min(keyboard_bounds[3], center_y + half_height),
    )
    if (
        not _valid_1000_bounds(snapped_bounds)
        or not _bounds_inside(snapped_bounds, keyboard_bounds, tolerance=0)
    ):
        return None
    return {
        "label": label or "↵",
        "bounds": [round(part) for part in snapped_bounds],
        "confidence": confidence,
        "key_action": key_action,
    }


def _preferred_keyboard_layout(character: str) -> str:
    if len(character) != 1:
        raise UISceneError("下一逐键字符必须恰好一个字符。")
    if character.isdecimal():
        return "numeric"
    if character == " " or character.isalpha():
        return "qwerty"
    return "symbol"


def _validated_keyboard_literal_keys(
    value: Any,
    *,
    keyboard_bounds: tuple[float, float, float, float] | None,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > 8:
        raise UISceneError("literal_keys 必须是最多8项的数组。")
    if value and keyboard_bounds is None:
        raise UISceneError("literal_keys 必须绑定完整可见键盘。")
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "value", "label", "key_kind", "bounds", "confidence", "fully_visible"
        }:
            raise UISceneError("literal_key 字段不符合协议。")
        key_value = item.get("value")
        label = item.get("label")
        key_kind = item.get("key_kind")
        if (
            not isinstance(key_value, str)
            or len(key_value) != 1
            or not isinstance(label, str)
            or key_kind not in {"character", "space"}
            or not _valid_1000_bounds(item.get("bounds"))
            or not isinstance(item.get("fully_visible"), bool)
        ):
            raise UISceneError("literal_key 内容无效。")
        if key_kind == "character" and (key_value == " " or label != key_value):
            raise UISceneError("字符键 label 必须逐字等于其输入值。")
        if key_kind == "space" and (
            key_value != " "
            or label.strip().casefold() not in {"", "space", "空格"}
        ):
            raise UISceneError("空格键缺少明确的空格语义。")
        bounds = tuple(float(part) for part in item["bounds"])
        confidence = _audit_confidence(item.get("confidence"), "literal_key")
        if (
            item["fully_visible"] is not True
            or confidence < 0.9
            or keyboard_bounds is None
            or not _bounds_inside(bounds, keyboard_bounds, tolerance=12)
        ):
            continue
        if key_kind == "space":
            keyboard_width = keyboard_bounds[2] - keyboard_bounds[0]
            keyboard_height = keyboard_bounds[3] - keyboard_bounds[1]
            if (
                bounds[2] - bounds[0] < 0.18 * keyboard_width
                or bounds[1] < keyboard_bounds[1] + 0.55 * keyboard_height
            ):
                continue
        result.append(
            {
                "value": key_value,
                "label": label,
                "key_kind": key_kind,
                "bounds": [round(part) for part in bounds],
                "confidence": confidence,
            }
        )
    return result


def _literal_key_overlaps_qwerty_letter_cell(
    item: dict[str, Any],
    *,
    qwerty_geometry: dict[str, Any],
) -> bool:
    """Reject alternate glyphs painted inside an alphabet key cell.

    The visual model supplies only the candidate box and seven row anchors.
    The existing local QWERTY validator deterministically reconstructs every
    alphabet-key center.  A non-letter literal candidate centered in one of
    those cells is therefore a secondary/long-press hint, not a direct key.
    A dedicated number row remains vertically separate and is unaffected.
    """

    bounds = item.get("bounds")
    anchors = qwerty_geometry.get("anchors")
    if (
        not isinstance(bounds, list)
        or len(bounds) != 4
        or not isinstance(anchors, dict)
    ):
        return True
    try:
        profile = qwerty_keyboard_config_from_anchors(anchors)
    except WorkflowNotReady:
        return True
    rows = profile.get("rows")
    if not isinstance(rows, list) or len(rows) != 3:
        return True
    row_y = [float(row["y"]) for row in rows]
    vertical_pitch = min(row_y[1] - row_y[0], row_y[2] - row_y[1])
    if vertical_pitch <= 0:
        return True
    center_x = (float(bounds[0]) + float(bounds[2])) / 2000.0
    center_y = (float(bounds[1]) + float(bounds[3])) / 2000.0
    for row in rows:
        keys = str(row.get("keys") or "")
        x_start = float(row["x_start"])
        x_step = float(row["x_step"])
        y = float(row["y"])
        if abs(center_y - y) > 0.45 * vertical_pitch:
            continue
        if any(
            abs(center_x - (x_start + index * x_step)) <= 0.48 * x_step
            for index in range(len(keys))
        ):
            return True
    return False


def _layout_switch_label_matches(label: str, target_layout: str) -> bool:
    normalized = label.strip().casefold()
    if target_layout == "numeric":
        return bool(re.search(r"(?:123|数字|num)", normalized))
    if target_layout == "qwerty":
        return bool(re.search(r"(?:abc|字母|英文|letters?)", normalized))
    if target_layout == "symbol":
        return bool(re.search(r"(?:符|sym|[#?+]=?|[.?]123)", normalized))
    return False


def _validated_keyboard_layout_switches(
    value: Any,
    *,
    keyboard_bounds: tuple[float, float, float, float] | None,
    current_layout: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > 4:
        raise UISceneError("layout_switches 必须是最多4项的数组。")
    if value and keyboard_bounds is None:
        raise UISceneError("layout_switches 必须绑定完整可见键盘。")
    result: list[dict[str, Any]] = []
    layouts = {"qwerty", "numeric", "symbol"}
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "label", "bounds", "confidence", "current_layout", "target_layout"
        }:
            raise UISceneError("layout_switch 字段不符合协议。")
        label = str(item.get("label") or "").strip()
        source = _normalized_keyboard_layout_token(
            item.get("current_layout")
        )
        target = _normalized_keyboard_layout_token(
            item.get("target_layout")
        )
        if (
            source not in layouts
            or target not in layouts
            or source == target
            or source != current_layout
            or not _valid_1000_bounds(item.get("bounds"))
        ):
            # Keep strict schema/type checks, but revoke an unsupported
            # optional direction instead of allowing it to veto an unrelated
            # valid literal key. No local switch element is minted from it.
            continue
        bounds = tuple(float(part) for part in item["bounds"])
        confidence = _audit_confidence(item.get("confidence"), "layout_switch")
        if (
            confidence < 0.9
            or not _layout_switch_label_matches(label, target)
            or keyboard_bounds is None
            or not _bounds_inside(bounds, keyboard_bounds, tolerance=12)
        ):
            continue
        result.append(
            {
                "label": label,
                "bounds": [round(part) for part in bounds],
                "confidence": confidence,
                "current_layout": source,
                "target_layout": target,
            }
        )
    return result


def _validated_keyboard_case_switch(
    value: Any,
    *,
    keyboard_bounds: tuple[float, float, float, float] | None,
    keyboard_layout: str,
    keyboard_input_mode: str,
    case_mode: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "label", "bounds", "confidence", "current_mode", "target_mode"
    }:
        raise UISceneError("case_switch 字段不符合协议。")
    current = value.get("current_mode")
    target = value.get("target_mode")
    label = str(value.get("label") or "").strip()
    if (
        keyboard_layout != "qwerty"
        or keyboard_input_mode != "direct_latin"
        or keyboard_bounds is None
        or current not in {"lower", "upper"}
        or target not in {"lower", "upper"}
        or current == target
        or current != case_mode
        or not _valid_1000_bounds(value.get("bounds"))
    ):
        raise UISceneError("case_switch 只允许绑定英文QWERTY明确大小写方向。")
    bounds = tuple(float(part) for part in value["bounds"])
    confidence = _audit_confidence(value.get("confidence"), "case_switch")
    if (
        confidence < 0.9
        or not re.search(r"(?:shift|大小写|大写|小写|⇧|↑|⬆)", label, re.IGNORECASE)
        or not _bounds_inside(bounds, keyboard_bounds, tolerance=12)
    ):
        return None
    return {
        "label": label,
        "bounds": [round(part) for part in bounds],
        "confidence": confidence,
        "current_mode": current,
        "target_mode": target,
    }


def _is_explicit_keyboard_mode_label(label: str) -> bool:
    visible = re.sub(r"[\s_\-/]+", "", str(label or "").strip().casefold())
    if not visible:
        return False
    if "中" in visible or "英" in visible:
        return True
    return visible in {
        "en",
        "eng",
        "english",
        "中文",
        "chinese",
        "abc",
        "latin",
        "pinyin",
    }


def _is_incomplete_optional_keyboard_mode_switch(value: Any) -> bool:
    """Recognize only a missing-field subset of the optional switch schema."""

    required = {
        "label",
        "bounds",
        "confidence",
        "current_mode",
        "target_mode",
    }
    return isinstance(value, dict) and set(value) < required


def _can_discard_separate_right_button(
    input_item: Any,
    value: Any,
) -> bool:
    """Discard a schema-valid control proven outside the input bounds."""

    required = {"label", "bounds", "confidence"}
    if (
        not isinstance(input_item, dict)
        or not isinstance(value, dict)
        or not set(value).issubset(required)
        or "bounds" not in value
        or not _valid_1000_bounds(input_item.get("bounds"))
        or not _valid_1000_bounds(value.get("bounds"))
    ):
        return False
    input_bounds = tuple(float(part) for part in input_item["bounds"])
    button_bounds = tuple(float(part) for part in value["bounds"])
    return bool(
        button_bounds[0] >= input_bounds[2] - 10
        and _vertical_overlap_ratio(button_bounds, input_bounds) >= 0.8
    )


def _input_structure_diagnostic_shape(raw: str) -> dict[str, Any]:
    """Return bounded field names and types without retaining observed text."""

    try:
        payload = _extract_json_object(raw)
    except Exception:
        return {"parseable": False}
    result: dict[str, Any] = {
        "parseable": True,
        "top_level_keys": sorted(str(key)[:80] for key in payload),
    }
    keyboard = payload.get("keyboard")
    result["keyboard_type"] = type(keyboard).__name__
    if not isinstance(keyboard, dict):
        return result
    result["keyboard_keys"] = sorted(str(key)[:80] for key in keyboard)
    mode_switch = keyboard.get("mode_switch")
    result["mode_switch_type"] = type(mode_switch).__name__
    if isinstance(mode_switch, dict):
        result["mode_switch_keys"] = sorted(str(key)[:80] for key in mode_switch)
        result["mode_switch_value_types"] = {
            str(key)[:80]: type(value).__name__
            for key, value in sorted(mode_switch.items(), key=lambda item: str(item[0]))
        }
    return result


def _has_any_semantic_term(item: dict[str, Any], terms: tuple[str, ...]) -> bool:
    visible = " ".join(
        str(item.get(key) or "") for key in ("meaning", "label")
    ).casefold()
    return any(term in visible for term in terms)


def _bounds_inside(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
    *,
    tolerance: float,
) -> bool:
    return (
        inner[0] >= outer[0] - tolerance
        and inner[1] >= outer[1] - tolerance
        and inner[2] <= outer[2] + tolerance
        and inner[3] <= outer[3] + tolerance
    )


def _vertical_overlap_ratio(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    overlap = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    smaller = min(first[3] - first[1], second[3] - second[1])
    return overlap / smaller if smaller > 0 else 0.0


def _bounds_overlap_ratio(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    """Return how much of ``first`` is covered by ``second``."""

    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    return width * height / first_area if first_area > 0 else 0.0


def _normalize_compact_scene_payload(payload: dict[str, Any]) -> None:
    """Normalize harmless compact-model shorthand before strict validation.

    The scene protocol still validates every field afterwards.  This only accepts
    the common JSON shorthand where a single evidence string is emitted instead
    of the requested one-item array; it does not repair coordinates, confidence,
    states or any action-like fields.
    """

    collection = payload.get("elements")
    if not isinstance(collection, list):
        return

    accepted: list[Any] = []
    for item in collection:
        if not isinstance(item, dict):
            accepted.append(item)
            continue

        evidence = item.get("evidence")
        if isinstance(evidence, str):
            text = evidence.strip()
            item["evidence"] = [text] if text else []

        role = str(item.get("role") or "").strip().lower()
        if role in ALLOWED_ROLES:
            accepted.append(item)
            continue

        states = item.get("states")
        goal_relevant = (
            isinstance(states, dict) and states.get("goal_relevant") is True
        )
        if goal_relevant:
            # Never coerce or discard an invalid target. Keeping it lets strict
            # protocol validation stop the controller before any action.
            accepted.append(item)
        # Unsupported peripheral structure is intentionally discarded. It is
        # not converted into a clickable role and therefore cannot be targeted.

    payload["elements"] = accepted


def _normalize_exact_target_ui_label_relevance(
    payload: dict[str, Any],
    goal_context: dict[str, Any],
) -> None:
    """Use one exact user-provided visible label to resolve model over-selection.

    This never invents an element or changes geometry.  A missing or duplicate
    exact label leaves the scene untouched so ambiguity remains fail-closed.
    """

    target_label = _goal_target_ui_label(goal_context)
    elements = payload.get("elements")
    if not target_label or not isinstance(elements, list):
        return
    matches = [
        item
        for item in elements
        if isinstance(item, dict)
        and str(item.get("label") or "").strip() == target_label
    ]
    if len(matches) != 1:
        return
    target = matches[0]
    for item in elements:
        if not isinstance(item, dict):
            continue
        states = item.get("states")
        if not isinstance(states, dict):
            continue
        states["goal_relevant"] = item is target
    # Exact text may resolve relevance, but never visibility authority.
    # ``fully_visible`` must be supplied by observation and independently
    # confirmed by the geometry audit before an element-bound action.


def _goal_target_ui_label(context: dict[str, Any]) -> str:
    """Return one consistent literal target label across graph context views."""

    sources: list[dict[str, Any]] = []
    focused = _active_subgoal_visual_context(context)
    if focused is not context:
        goal_entities = focused.get("goal_entities")
        if isinstance(goal_entities, dict):
            sources.append(goal_entities)
    entities = context.get("entities")
    if isinstance(entities, dict):
        sources.append(entities)
    labels = {
        str(source.get("target_ui_label") or "").strip()
        for source in sources
        if str(source.get("target_ui_label") or "").strip()
    }
    return next(iter(labels)) if len(labels) == 1 else ""


def _strip_model_authored_local_attestations(payload: dict[str, Any]) -> None:
    """Private controller facts can only be minted by local audit code."""

    elements = payload.get("elements")
    if not isinstance(elements, list):
        return
    for item in elements:
        if not isinstance(item, dict) or not isinstance(item.get("states"), dict):
            continue
        item["states"].pop("reload_visual_audit", None)
        item["states"].pop("independent_geometry_verified", None)
        item["states"].pop("geometry_audit_source", None)
        states = item["states"]
        if states.get("keyboard_input_mode_switch") is True:
            modes = {"direct_latin", "chinese_pinyin"}
            current_mode = states.get("current_mode")
            target_mode = states.get("target_mode")
            if (
                current_mode not in modes
                or target_mode not in modes
                or current_mode == target_mode
            ):
                # Revoke an incomplete model-authored permission claim.  A
                # later local input-structure audit may mint a directional
                # switch again from the same pixels; this normalization never
                # creates a target or guesses the missing direction.
                states.pop("keyboard_input_mode_switch", None)
                states.pop("current_mode", None)
                states.pop("target_mode", None)


def _normalize_non_target_keyboard_switch(
    payload: dict[str, Any],
    goal_context: dict[str, Any],
) -> None:
    """Discard keyboard-switch candidates only for goals unrelated to input.

    An unrelated peripheral switch cannot help complete a refresh, navigation,
    or content goal. Removing it cannot authorize an action, while retaining a
    malformed direction can block every other valid candidate. Input and
    keyboard-mode goals keep the element for strict validation and planning.
    Any action-like or protocol-extra field also keeps the element so the scene
    parser fails closed instead of hiding it.
    """

    if _goal_requests_input(goal_context) or _goal_requests_keyboard_mode_switch(
        goal_context
    ):
        return
    elements = payload.get("elements")
    if not isinstance(elements, list):
        return
    exact_fields = {
        "element_id",
        "role",
        "meaning",
        "label",
        "bounds",
        "confidence",
        "states",
        "evidence",
    }
    action_like = {
        "action",
        "actions",
        "plan",
        "step",
        "steps",
        "tap",
        "swipe",
        "command",
        "coordinates",
    }

    def contains_action_like_key(value: Any) -> bool:
        if isinstance(value, dict):
            return any(
                str(key).strip().casefold() in action_like
                or contains_action_like_key(part)
                for key, part in value.items()
            )
        if isinstance(value, list):
            return any(contains_action_like_key(part) for part in value)
        return False

    kept: list[Any] = []
    for item in elements:
        if not isinstance(item, dict):
            kept.append(item)
            continue
        states = item.get("states")
        claimed_switch = (
            str(item.get("meaning") or "").strip() == "switch_keyboard_input_mode"
            or (
                isinstance(states, dict)
                and states.get("keyboard_input_mode_switch") is True
            )
        )
        safely_discardable = (
            claimed_switch
            and set(item) == exact_fields
            and not contains_action_like_key(item)
        )
        if not safely_discardable:
            kept.append(item)
    payload["elements"] = kept


def _normalize_reload_goal_safety(
    payload: dict[str, Any],
    goal_context: dict[str, Any],
) -> None:
    """For reload goals, only a literal refresh/reload control may be a target.

    This only revokes model-reported relevance. It never creates an element,
    changes geometry, or treats static page content as proof that a reload
    event happened. With no valid reload candidate, the observer's existing
    targeted-refinement path can inspect an explicitly requested visual region.
    """

    if not _goal_requests_reload(goal_context):
        return
    elements = payload.get("elements")
    if not isinstance(elements, list):
        return
    reload_terms = {
        "refresh",
        "reload",
        "refresh_page",
        "reload_page",
        "刷新",
        "重新加载",
        "刷新页面",
    }
    for item in elements:
        if not isinstance(item, dict):
            continue
        states = item.get("states")
        if not isinstance(states, dict):
            continue
        meaning = str(item.get("meaning") or "").strip().casefold()
        label = str(item.get("label") or "").strip().casefold()
        is_reload_control = meaning in reload_terms or label in reload_terms
        states["goal_relevant"] = bool(is_reload_control)


def _normalize_known_scene_enums(payload: dict[str, Any]) -> None:
    """Normalize only casing/outer whitespace for already-known enum tokens.

    Unknown values are intentionally preserved so the strict UI scene parser
    still rejects them instead of guessing a keyboard fact.
    """

    elements = payload.get("elements")
    if not isinstance(elements, list):
        return
    enum_fields = {
        "keyboard_layout": {"qwerty", "numeric", "symbol", "unknown"},
        "keyboard_input_mode": {
            "direct_latin",
            "chinese_pinyin",
            "unknown",
        },
        "current_mode": {"direct_latin", "chinese_pinyin"},
        "target_mode": {"direct_latin", "chinese_pinyin"},
    }
    explicit_mode_aliases = {
        "english": "direct_latin",
        "chinese": "chinese_pinyin",
    }
    for item in elements:
        if not isinstance(item, dict):
            continue
        states = item.get("states")
        if not isinstance(states, dict):
            continue
        role = str(item.get("role") or "").strip()
        if role == "keyboard_key":
            # A regular key can never own whole-keyboard layout or input-mode
            # facts and is excluded from the semantic action surface. Revoking
            # these misplaced fields cannot authorize the key or create a new
            # target, even when the model overstates its goal relevance.
            states.pop("keyboard_layout", None)
            states.pop("keyboard_input_mode", None)
        elif role != "input" and states.get("goal_relevant") is not True:
            # Qwen sometimes emits a non-interactive keyboard container and
            # attaches global keyboard facts to it. Those peripheral facts are
            # not actionable and the scene protocol intentionally allows them
            # only on the bound input. Remove them only from non-target
            # elements; a malformed goal element must still fail closed.
            states.pop("keyboard_layout", None)
            states.pop("keyboard_input_mode", None)
        for field, allowed in enum_fields.items():
            value = states.get(field)
            if not isinstance(value, str):
                continue
            normalized = value.strip().casefold()
            if field == "keyboard_layout":
                normalized = _normalized_keyboard_layout_token(normalized)
            if field in {
                "keyboard_input_mode",
                "current_mode",
                "target_mode",
            }:
                normalized = explicit_mode_aliases.get(normalized, normalized)
            if normalized in allowed:
                states[field] = normalized


def _scene_enum_values(raw: str) -> dict[str, list[str]]:
    """Return only keyboard enum tokens for safe local failure diagnostics."""

    try:
        payload = _extract_json_object(raw)
    except Exception:
        return {}
    collected: dict[str, set[str]] = {
        "keyboard_layout": set(),
        "keyboard_input_mode": set(),
        "current_mode": set(),
        "target_mode": set(),
    }

    def remember(container: Any, source_key: str, target_key: str) -> None:
        if not isinstance(container, dict) or source_key not in container:
            return
        value = container.get(source_key)
        rendered = str(value).strip()[:80]
        if rendered:
            collected[target_key].add(rendered)

    elements = payload.get("elements")
    if isinstance(elements, list):
        for item in elements:
            if not isinstance(item, dict):
                continue
            states = item.get("states")
            for key in tuple(collected):
                remember(states, key, key)

    keyboard = payload.get("keyboard")
    remember(keyboard, "layout", "keyboard_layout")
    remember(keyboard, "input_mode", "keyboard_input_mode")
    mode_switch = keyboard.get("mode_switch") if isinstance(keyboard, dict) else None
    remember(mode_switch, "current_mode", "current_mode")
    remember(mode_switch, "target_mode", "target_mode")
    return {
        key: sorted(values)
        for key, values in collected.items()
        if values
    }


def _goal_requests_coordinate_free_system_home(
    context: dict[str, Any],
) -> bool:
    """Recognize only an explicit active system-Home transition.

    App-local labels such as ``主页`` or ``首页`` deliberately do not match.
    The completion condition is required as a second typed graph signal so a
    compound root goal mentioning a later return cannot suppress refinement
    for its current element-bound step.
    """

    focused = _observation_goal_context(context)
    if str(focused.get("execution_class") or "").strip() != "navigate":
        return False
    objective = re.sub(
        r"\s+",
        "",
        str(focused.get("objective") or "").strip().casefold(),
    )
    conditions = focused.get("completion_conditions")
    if not objective or not isinstance(conditions, list):
        return False
    objective_matches = bool(
        re.search(
            r"(?:返回|回到|退回|切回)(?:手机|设备)?(?:的)?(?:桌面|主屏幕)",
            objective,
        )
        or re.search(
            r"\b(?:return|go|switch)(?:back)?to(?:the)?(?:phone|device)?homescreen\b",
            objective,
        )
    )
    if not objective_matches:
        return False
    return any(
        re.search(
            r"(?:手机|设备)?(?:的)?(?:桌面|主屏幕)(?:已)?(?:可见|显示|在前台)",
            re.sub(r"\s+", "", str(condition or "").strip().casefold()),
        )
        for condition in conditions
    )


def _is_verified_navigation_result_observation(
    context: dict[str, Any],
) -> bool:
    """Recognize the adapter-owned observation phase after safe navigation.

    This marker carries no execution authority.  It only prevents a second
    target-refinement pass from searching the destination page for the source
    control that was just consumed.  The exact locally generated objective and
    completion condition are required in addition to the typed graph impact.
    """

    focused = _observation_goal_context(context)
    if str(focused.get("execution_class") or "").strip() != "navigate":
        return False
    entities = focused.get("goal_entities")
    if not isinstance(entities, dict):
        return False
    return (
        entities.get("observation_phase")
        == POST_NAVIGATION_RESULT_OBSERVATION_PHASE
        and str(focused.get("objective") or "").strip()
        == POST_NAVIGATION_RESULT_OBJECTIVE
        and focused.get("completion_conditions")
        == POST_NAVIGATION_RESULT_COMPLETION_CONDITIONS
    )


def _needs_targeted_refinement(scene: UIScene, context: dict[str, Any]) -> bool:
    if not context:
        return False
    focused = _observation_goal_context(context)
    if _is_verified_navigation_result_observation(context):
        return False
    if _goal_requests_coordinate_free_system_home(context):
        # System Home has no element geometry.  Asking the target refiner to
        # find a control can conflate an App-local ``主页`` tab with the device
        # Home action.  The compact scene, Qwen system-action proposal and the
        # normal controller/post-observation gates remain authoritative.
        return False
    if scene.confidence < 0.72:
        return True
    if _goal_requests_page_title(context) and not _scene_has_grounded_page_title(scene):
        return True
    target_label = _goal_target_ui_label(context)
    if target_label:
        exact_matches = [
            element for element in scene.elements if element.label == target_label
        ]
        if len(exact_matches) != 1:
            return True
        if not any(item.strip() for item in exact_matches[0].evidence):
            return True
    goal_elements = [
        element
        for element in scene.elements
        if element.states.get("goal_relevant") is True
    ]
    if any(
        element.confidence >= 0.72
        and element.states.get("fully_visible") is not False
        and any(item.strip() for item in element.evidence)
        for element in goal_elements
    ):
        return False
    if goal_elements:
        # A compact pass may notice only a clipped or low-confidence target.
        # That is evidence to inspect more closely, never evidence to act on or
        # a reason to skip the existing goal-directed refinement. Input goals
        # keep their stricter dedicated structure audit instead of spending
        # this generic refinement on a clipped field shell.
        return not _goal_requests_input(context)
    if scene.screen_id == "unknown":
        return True

    target_app = str(context.get("app_id") or "").strip().casefold()
    objective = str(focused.get("objective") or "").strip()
    if target_app and re.search(r"^(打开|进入|启动)", objective):
        # Seeing an App name somewhere on Launcher is not an actionable target.
        # When no trusted goal element survived above, a not-yet-foreground App
        # requires a focused pass to establish one complete, relevant entry.
        return scene.foreground_app_id.casefold() != target_app

    terms = _goal_terms(context)
    if not terms:
        return False
    visible = " ".join(
        [scene.summary]
        + [
            " ".join([element.label, element.meaning, *element.evidence])
            for element in scene.elements
        ]
    ).casefold()
    return not any(term in visible for term in terms)


def _goal_terms(context: dict[str, Any]) -> tuple[str, ...]:
    focused = _observation_goal_context(context)
    values: list[str] = []
    for key in ("app_id", "app_name"):
        value = str(context.get(key) or "").strip().casefold()
        if value:
            values.append(value)
    entities = (
        focused.get("goal_entities")
        if focused is not context
        else context.get("entities")
    )
    if isinstance(entities, dict):
        values.extend(str(value).strip().casefold() for value in entities.values())
    objective = str(focused.get("objective") or "").strip().casefold()
    if objective:
        simplified = re.sub(
            r"打开|进入|启动|点击|选择|查找|搜索|关闭|返回|当前|页面|应用|app|然后|请|帮我",
            " ",
            objective,
        )
        values.extend(re.findall(r"[a-z0-9_]{1,40}|[\u4e00-\u9fff]{1,12}", simplified))
        values.extend(re.findall(r"[0-9]+", objective))
    return tuple(dict.fromkeys(value for value in values if value))


def _local_frame_fingerprint(frame: Image.Image) -> str:
    compact = frame.convert("L").resize((64, 96), Image.Resampling.BILINEAR)
    return hashlib.sha256(compact.tobytes()).hexdigest()[:20]


def _observation_cache_key(
    *,
    device_id: str | None,
    fingerprint: str,
    goal_context: dict[str, Any],
    input_lineage: TypedInputLineage | None,
) -> str | None:
    resolved_device = str(device_id or "").strip()
    if resolved_device.casefold() in {"", "unbound", "unknown", "none", "null"}:
        return None
    lineage_payload = (
        input_lineage.to_dict() if input_lineage is not None else None
    )
    payload = {
        "device_id": resolved_device,
        "fingerprint": str(fingerprint or "").strip(),
        "goal_context": goal_context,
        "input_lineage": lineage_payload,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _typed_input_continuation_base_scene(
    *,
    prior_scene: UIScene | None,
    input_lineage: TypedInputLineage | None,
    device_id: str | None,
    goal_context: dict[str, Any],
    fingerprint: str,
) -> UIScene | None:
    """Carry only prior surface identity into one pending typed input audit.

    The pending receipt lineage proves which already-authorized input
    transaction produced the new frame.  All prior elements, values,
    candidates and geometry are removed; the fresh dedicated input audit must
    reconstruct them.  This replaces a duplicate compact model call without
    granting stale geometry or a second input-state authority.
    """

    if prior_scene is None or input_lineage is None:
        return None
    resolved_device = str(device_id or "").strip()
    active_field_id = _goal_active_input_field(goal_context)[0]
    if (
        not resolved_device
        or not active_field_id
        or input_lineage.source not in PENDING_INPUT_LINEAGE_SOURCES
        or not _goal_requests_input(goal_context)
        or input_lineage.before_fingerprint != prior_scene.fingerprint
    ):
        return None
    try:
        prior_scene.validate()
        input_lineage.validate()
    except (UISceneError, ValueError):
        return None
    if not input_lineage.matches_typed_context(
        device_id=resolved_device,
        app_id=prior_scene.app_id,
        screen_id=prior_scene.screen_id,
        input_field_id=active_field_id,
    ):
        return None
    prior_inputs = tuple(
        element
        for element in prior_scene.elements
        if element.role == "input"
        and element.meaning == "application_text_input"
        and str(element.states.get("input_field_id") or "").strip()
        == active_field_id
        and element.states.get("fully_visible") is True
    )
    if len(prior_inputs) != 1:
        return None
    prior_input = prior_inputs[0]
    if any(
        abs(float(left) - float(right)) > 0.08
        for left, right in zip(prior_input.bounds, input_lineage.input_bounds)
    ):
        return None
    continued = replace(
        prior_scene,
        summary=(
            "沿用同一typed输入事务已验证的App/screen身份；"
            "当前字段、正文、候选和键盘全部等待新专用审计。"
        ),
        elements=(),
        overlays=(),
        stable=True,
        fingerprint=fingerprint,
    )
    continued.validate()
    return continued


def _safe_goal_context(value: dict[str, Any]) -> dict[str, Any]:
    """Keep goal data useful to OCR while refusing hidden control instructions."""

    forbidden = {
        "action",
        "actions",
        "step",
        "steps",
        "tap",
        "swipe",
        "coordinate",
        "coordinates",
        "x",
        "y",
        "command",
        "shell",
        "execution_plan",
    }

    def clean(item: Any, depth: int = 0) -> Any:
        if depth > 5:
            raise VisionAgentError("目标上下文嵌套过深。")
        if isinstance(item, dict):
            result: dict[str, Any] = {}
            for raw_key, raw_value in item.items():
                key = str(raw_key).strip()
                if key.lower() in forbidden:
                    raise VisionAgentError(f"目标上下文包含控制字段：{key}")
                result[key[:80]] = clean(raw_value, depth + 1)
            return result
        if isinstance(item, (list, tuple)):
            return [clean(part, depth + 1) for part in list(item)[:50]]
        if isinstance(item, str):
            return item[:1000]
        if isinstance(item, (int, float, bool)) or item is None:
            return item
        raise VisionAgentError("目标上下文包含不支持的数据类型。")

    cleaned = clean(value)
    if not isinstance(cleaned, dict):
        raise VisionAgentError("目标上下文必须是对象。")
    return cleaned
