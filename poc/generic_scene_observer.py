from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import threading
import time
from dataclasses import replace
from typing import Any, Iterator

from PIL import Image, ImageChops, ImageFilter

from element_geometry_audit import (
    ElementGeometryAuditError,
    build_candidate_crop_transform,
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
    CameraAlignmentFacts,
    MIN_TARGET_CONFIDENCE,
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


GENERIC_SCENE_OBSERVER_VERSION = "2026-08-19-generic-scene-observer-v58"
TARGETED_SCENE_DELTA_PROTOCOL_VERSION = "2026-08-17-targeted-scene-delta-v1"
FOREGROUND_APP_IDENTITY_AUDIT_VERSION = (
    "2026-08-18-foreground-app-identity-audit-v1"
)
INPUT_STRUCTURE_AUDIT_VERSION = "2026-08-18-input-structure-audit-v7"
SYSTEM_UI_AUDIT_VERSION = "2026-08-14-system-ui-audit-v1"
ICON_CLUSTER_AUDIT_VERSION = "2026-08-15-icon-cluster-audit-v1"
COMPACT_OUTPUT_TOKENS = 2600
# Targeted delta permits the same maximum element count as compact observation.
# Its output budget must therefore cover the same strict worst-case structure.
TARGETED_OUTPUT_TOKENS = COMPACT_OUTPUT_TOKENS
FOREGROUND_APP_IDENTITY_AUDIT_TOKENS = 300
INPUT_STRUCTURE_AUDIT_TOKENS = 1000
SYSTEM_UI_AUDIT_TOKENS = 600
ICON_CLUSTER_AUDIT_TOKENS = 700
ELEMENT_GEOMETRY_AUDIT_TOKENS = 500
ORIENTATION_AUDIT_TOKENS = 500
MIN_SYSTEM_UI_AUDIT_CONFIDENCE = 0.80
OBSERVATION_TIMEOUT_SECONDS = 60.0
MAX_COMPACT_ELEMENTS = 12
AUDITED_SOFT_KEYBOARD_HIDDEN_EVIDENCE = "输入结构只读审计确认软键盘不可见"

STAGE_LABELS = {
    "idle": "空闲",
    "checking_stability": "检查画面稳定性",
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

    def __init__(self, provider: Any) -> None:
        self.provider = provider
        self.last_raw_response = ""
        self.last_diagnostics: dict[str, Any] = {}
        self._stage_lock = threading.RLock()
        self._current_stage = "idle"
        self._last_stage = "idle"
        self.last_orientation_audit_diagnostics: dict[str, Any] = {}
        self.last_geometry_audit_diagnostics: dict[str, Any] = {}

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
            transform = (
                build_literal_candidate_crop_transform(frame.size, element.bounds)
                if literal_selector
                else build_candidate_crop_transform(frame.size, element.bounds)
            )
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
                    "crop_profile": (
                        "literal_selector" if literal_selector else "broad_structural"
                    ),
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

        images = (
            frame,
            frame.transpose(Image.Transpose.ROTATE_90),
            frame.transpose(Image.Transpose.ROTATE_270),
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
            raw = self._provider_chat(
                [_json_only_system_message(), {"role": "user", "content": content}],
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
            system_ui_audit_required = _goal_requests_system_ui_audit(context)
            camera_layout_orientation = _camera_layout_orientation(frame)
            image_part = {
                "type": "image_url",
                "image_url": {"url": _image_data_url(frame)},
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

            def parse_compact_response(value: str) -> UIScene:
                nonlocal compact_geometry_discarded, compact_input_geometry_isolated
                payload = _extract_compact_json_object(value)
                compact_input_geometry_isolated = (
                    _strip_preliminary_input_geometry_for_dedicated_audit(
                        payload,
                        context,
                    )
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

            if (
                not system_ui_audit_required
                and not _goal_requests_keyboard_mode_switch(context)
                and not compact_input_geometry_isolated
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

            if _needs_foreground_app_identity_audit(scene, context):
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

            if _goal_requests_reload(context):
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

            if system_ui_audit_required:
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
                self._set_stage("waiting_input_structure_audit")
                audit_content: list[dict[str, Any]] = [
                    {
                        "type": "text",
                        "text": _input_structure_audit_prompt(
                            context,
                            roi_bounds=None,
                        ),
                    },
                    image_part,
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
                        ),
                        visual_obstructions,
                        fingerprint=fingerprint,
                    )
                    input_retry_roi = _goal_directed_roi_bounds(context)
                    if (
                        _goal_has_explicit_input_text(context)
                        and input_retry_roi is not None
                        and not _input_audit_established_local_target(scene)
                    ):
                        # A valid empty audit grants no geometry authority. One
                        # independent single-crop retry is allowed only when the
                        # active goal itself supplies a coarse spatial region.
                        # The crop owns a local 0..1000 coordinate system and is
                        # mapped back locally; the first empty result contributes
                        # no fields, bounds or states. Two empty results still
                        # leave the scene fail-closed.
                        input_structure_audit_retry_used = True
                        self._set_stage("waiting_input_structure_audit")
                        retry_content = [
                            {
                                "type": "text",
                                "text": _input_structure_audit_prompt(
                                    context,
                                    roi_bounds=input_retry_roi,
                                    crop_local=True,
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
                            ),
                            visual_obstructions,
                            fingerprint=fingerprint,
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
                "output_token_budget": model_call_token_budgets[-1],
            }
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
                "max_attempts": 2,
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
_MIN_FOREGROUND_APP_IDENTITY_CONFIDENCE = 0.90


def _is_foreground_app_identity_placeholder(value: str) -> bool:
    return str(value or "").strip().casefold() in (
        _FOREGROUND_APP_IDENTITY_PLACEHOLDERS
    )


def _needs_foreground_app_identity_audit(
    scene: UIScene,
    context: dict[str, Any],
) -> bool:
    """Audit unknown identity only when a real named App is task-bound."""

    foreground = str(scene.foreground_app_id or "").strip().casefold()
    if _is_foreground_app_identity_placeholder(foreground):
        return True
    if foreground != "unknown":
        return False
    target_app = str(context.get("app_id") or "").strip().casefold()
    return bool(
        target_app
        and target_app != "unknown"
        and not _is_foreground_app_identity_placeholder(target_app)
    )


def _foreground_app_identity_audit_prompt() -> str:
    return f"""
You are an app-independent, read-only foreground application identity auditor.
Inspect only the physical phone display in this one stable image. No user goal,
target App, planned action, or previous model answer is provided or authoritative.

Return a short lower_snake_case semantic category for the App that is visibly in
the foreground. Use "unknown" when the visible chrome and content do not establish
one category with high confidence. Never return a referential placeholder such as
current_foreground, current_app, foreground_app, target_app, or active_app.

Evidence must contain one or two short visible identity cues from the phone screen.
Do not mention coordinates, bounds, PX/MM, robot controls, calibration, or any tap,
press, swipe, drag, execution, or suggestion. This audit grants no action authority
and must not describe a workflow.

Return exactly one JSON object with no Markdown, duplicate keys, or extra fields:
{{"protocol_version":"{FOREGROUND_APP_IDENTITY_AUDIT_VERSION}",
"foreground_app_id":"unknown","confidence":0.0,
"evidence":["short visible App identity cue"]}}
"""


def _compact_prompt(context: dict[str, Any]) -> str:
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


def _input_structure_audit_prompt(
    context: dict[str, Any],
    *,
    roi_bounds: tuple[int, int, int, int] | None,
    crop_local: bool = False,
) -> str:
    context = _observation_goal_context(context)
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
            "coordinates. If a structure cannot be bounded in this crop-local "
            "coordinate system, omit it instead of clipping or converting it."
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
            "height. If a structure cannot be bounded in this coordinate "
            "system, omit it instead of clipping or converting it."
        )
    return f"""
You are a read-only, app-independent UI structure auditor. The normal scene observer did not establish an input target.
Goal context (evidence selection only): {json.dumps(context, ensure_ascii=False, separators=(',', ':'))}
{image_contract}
Distinguish three different visual structures; never merge them:
1. application_inputs: editable search/address/form fields in the App content area. Include an empty field only when a complete border plus a visible placeholder, caret, focus highlight, or other literal editable cue is visible.
2. ime_preedit_regions: the input method's composition/candidate strip. It is never an application input, even when it contains composed text and a trailing icon. Enumerate only complete visible candidate words inside each region; candidates are read-only facts and never application inputs.
3. keyboard.mode_switch: one compact key inside the visible keyboard that explicitly switches between chinese_pinyin and direct_latin. Ordinary letters, backspace, enter, robot/assistant, voice, emoji, and candidate-strip icons are never mode switches.
4. keyboard.qwerty_anchors: only for a complete visible QWERTY keyboard, locate the centers of q, p, a, l, z, m and backspace. These are read-only current-frame geometry facts, not a tap plan. Use null for every non-QWERTY, incomplete or uncertain keyboard.
5. keyboard.backspace_key: for any complete visible keyboard layout, report the one complete backspace/delete key as label, bounds, confidence and fully_visible. Use null when absent, clipped, ambiguous, or confused with an App delete control. This is read-only geometry and never authorizes clearing by itself.
6. keyboard.literal_keys: enumerate at most eight complete visible keys that insert exactly one character. value is the exact inserted character; for the space bar use value=" " and key_kind="space". For every other key use key_kind="character" and require label to equal value literally. Never include backspace, enter, send/search, emoji, voice, assistant, shift, or layout switches.
6. keyboard.layout_switches: enumerate only compact visible keys with an explicit destination layout: qwerty, numeric, or symbol. Copy the literal label and report current_layout and target_layout; never infer a destination from the goal alone.
7. keyboard.case_mode and keyboard.case_switch apply only to direct_latin QWERTY. case_mode is lower, upper, or unknown from the visible letter glyphs. case_switch is null unless a complete visible shift/case key and its lower↔upper direction are independently clear.
Determine keyboard.input_mode only from the current whole keyboard image, never from the goal or the JSON example. Visible Chinese composition/candidates, pinyin separators, or a current-mode label such as 中/中文/Pinyin prove chinese_pinyin. A visible current-mode label such as 英/EN/English/ABC/Latin together with a plain Latin QWERTY layout and no Chinese composition/candidate strip proves direct_latin. If the whole keyboard does not prove the current mode, use unknown and set mode_switch to null.
keyboard.mode_switch.current_mode MUST equal keyboard.input_mode whenever input_mode is known. Treat an unambiguous single-mode label on the key as the current visible mode: 中/中文/Pinyin means chinese_pinyin; 英/EN/English/ABC/Latin means direct_latin. If the label could instead name a destination and the current whole-keyboard state is not independently clear, do not guess a direction; set mode_switch to null.
For a text-entry verification goal, report the proven current keyboard.input_mode; keyboard.mode_switch is optional and should be null unless its direction is independently unambiguous. Never invent a switch direction merely because the goal asks for text entry.
keyboard.mode_switch MUST be either null or an object with exactly these five fields: label, bounds, confidence, current_mode, target_mode. Never omit confidence or target_mode. Valid non-null shapes in the two directions are:
{{"label":"中","bounds":[0,0,1000,1000],"confidence":0.0,"current_mode":"chinese_pinyin","target_mode":"direct_latin"}}
{{"label":"英","bounds":[0,0,1000,1000],"confidence":0.0,"current_mode":"direct_latin","target_mode":"chinese_pinyin"}}
These are shape examples only. Copy the literal visible label and measured bounds from Image 1, set confidence from the visible evidence, and choose the direction from the independently proven current keyboard state. Never copy either example merely to satisfy the goal.
keyboard.case_switch uses the same five field names, but current_mode and target_mode are lower or upper. It is valid only for direct_latin QWERTY and a visible shift/case glyph. Example shape: {{"label":"⇧","bounds":[0,0,1000,1000],"confidence":0.0,"current_mode":"lower","target_mode":"upper"}}.
Do not plan, suggest, authorize, or perform any action.
{coordinate_contract}
Use text="" for a visibly empty application field. Copy placeholders and visible_editable_cues literally; do not infer them from the goal. right_button describes a trailing utility control; it is structural evidence only and is never authorized for activation. Set it to null when no separate trailing control is visible.
Return exactly this JSON schema and no other fields:
{{"protocol_version":"{INPUT_STRUCTURE_AUDIT_VERSION}",
"application_inputs":[{{"structure_id":"app-input-1","bounds":[0,0,1000,1000],
"fully_visible":true,"text":"","placeholder":"visible placeholder or empty",
"visible_editable_cues":["literal visible cue"],"confidence":0.0,
"right_button":null}}],
"ime_preedit_regions":[{{"region_id":"ime-preedit-1","bounds":[0,0,1000,1000],
"text":"visible composition text or empty","confidence":0.0,
"candidates":[{{"text":"literal candidate","bounds":[0,0,1000,1000],"confidence":0.0,"fully_visible":true}}]}}],
"keyboard":{{"visible":true,"bounds":[0,0,1000,1000],"layout":"qwerty",
"input_mode":"unknown","case_mode":"unknown","qwerty_anchors":{{"q":[0,0],"p":[0,0],"a":[0,0],"l":[0,0],"z":[0,0],"m":[0,0],"backspace":[0,0]}},"mode_switch":null,
"backspace_key":{{"label":"⌫","bounds":[0,0,1000,1000],"confidence":0.0,"fully_visible":true}},
"case_switch":null,"literal_keys":[{{"value":".","label":".","key_kind":"character","bounds":[0,0,1000,1000],"confidence":0.0,"fully_visible":true}}],
"layout_switches":[{{"label":"123","bounds":[0,0,1000,1000],"confidence":0.0,"current_layout":"qwerty","target_layout":"numeric"}}]}}}}
When no keyboard is visible, keyboard must be {{"visible":false,"bounds":null,"layout":"unknown","input_mode":"unknown","case_mode":"unknown","qwerty_anchors":null,"mode_switch":null,"backspace_key":null,"case_switch":null,"literal_keys":[],"layout_switches":[]}}.
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
    if (
        not isinstance(evidence, list)
        or not 1 <= len(evidence) <= 2
        or any(
            not camera_alignment_evidence_is_safe(item)
            or len(str(item).strip()) > 120
            for item in evidence
        )
    ):
        raise VisionAgentError("前台应用身份审计 evidence 不安全或格式无效。")
    evidence_tuple = tuple(str(item).strip() for item in evidence)
    if app_id == "unknown" or confidence < _MIN_FOREGROUND_APP_IDENTITY_CONFIDENCE:
        app_id = "unknown"
    return app_id, confidence, evidence_tuple


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
    """Normalize only an exact, finite crop-scale x/y/w/h bounds object."""

    if not isinstance(payload, dict) or not isinstance(payload.get("elements"), list):
        return
    for element in payload["elements"]:
        if not isinstance(element, dict):
            continue
        bounds = element.get("bounds")
        if not isinstance(bounds, dict) or set(bounds) != {"x", "y", "w", "h"}:
            continue
        values = tuple(bounds[key] for key in ("x", "y", "w", "h"))
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
        _strip_model_authored_local_attestations(payload)
        _normalize_non_target_keyboard_switch(payload, goal_context or {})
        _normalize_reload_goal_safety(payload, goal_context or {})
        _normalize_known_scene_enums(payload)
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
    """Remove unsafe peripheral evidence only when safe phone evidence remains."""

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
        "external_impact",
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
        "external_impact": focused["external_impact"],
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
    visible = json.dumps(
        _active_subgoal_visual_context(context),
        ensure_ascii=False,
    ).casefold()
    return any(
        term in visible
        for term in (
            "切换输入模式",
            "切换到英文",
            "切到英文",
            "切换到中文",
            "切到中文",
            "切换直输模式",
            "切换为直输模式",
            "switch input mode",
            "switch keyboard mode",
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

    candidate = scene.unique_trusted_goal_element()
    return bool(
        candidate is not None
        and candidate.element_id
        in {
            "local_audited_input_1",
            "local_audited_keyboard_mode_switch_1",
            "local_audited_ime_candidate_1",
        }
    )


def _goal_has_explicit_input_text(context: dict[str, Any]) -> bool:
    """Return true only when the graph supplied a concrete text-entry entity."""

    focused = _active_subgoal_visual_context(context)
    if focused is context:
        entities = context.get("entities")
    else:
        entities = focused.get("goal_entities")
    return bool(
        isinstance(entities, dict)
        and isinstance(entities.get("input_text"), str)
        and entities["input_text"].strip()
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


def _apply_input_structure_audit(
    scene: UIScene,
    raw: str,
    *,
    fingerprint: str,
    goal_context: dict[str, Any],
) -> UIScene:
    try:
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
            normalized_layout = keyboard_layout.strip().casefold()
            if normalized_layout == "symbols":
                normalized_layout = "symbol"
            if normalized_layout in {"qwerty", "numeric", "symbol", "unknown"}:
                keyboard_layout = normalized_layout
                keyboard["layout"] = normalized_layout
        keyboard_input_mode = keyboard.get("input_mode")
        keyboard_case_mode = keyboard.get("case_mode", "unknown")
        if not isinstance(keyboard_visible, bool):
            raise UISceneError("输入结构审计 keyboard.visible 必须是布尔值。")
        if keyboard_layout not in {"qwerty", "numeric", "symbol", "unknown"}:
            raise UISceneError("输入结构审计 keyboard.layout 无效。")
        if keyboard_input_mode not in {
            "direct_latin",
            "chinese_pinyin",
            "unknown",
        }:
            raise UISceneError("输入结构审计 keyboard.input_mode 无效。")
        if keyboard_case_mode not in {"lower", "upper", "unknown"}:
            raise UISceneError("输入结构审计 keyboard.case_mode 无效。")
        keyboard_bounds: tuple[float, float, float, float] | None = None
        boundsless_keyboard_dismissal = False
        if keyboard_visible:
            valid_keyboard_bounds = _valid_1000_bounds(keyboard.get("bounds"))
            if valid_keyboard_bounds:
                keyboard_bounds = tuple(float(value) for value in keyboard["bounds"])
                valid_keyboard_bounds = bool(
                    keyboard_bounds[2] - keyboard_bounds[0] >= 300
                    and keyboard_bounds[3] - keyboard_bounds[1] >= 180
                )
            if not valid_keyboard_bounds:
                if not (
                    _goal_requests_keyboard_dismissal(goal_context)
                    and _scene_reports_keyboard(scene)
                ):
                    raise UISceneError("可见键盘必须提供有效 bounds。")
                # A dismissal-only subgoal can safely authorize Android Back
                # without touching the keyboard.  Keep only dual-source
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
                keyboard["case_switch"] = None
                keyboard["literal_keys"] = []
                keyboard["layout_switches"] = []
                keyboard.pop("qwerty_anchors", None)
        elif (
            keyboard.get("bounds") is not None
            or keyboard.get("mode_switch") is not None
            or keyboard.get("backspace_key") is not None
            or keyboard.get("case_switch") is not None
            or keyboard.get("literal_keys") not in (None, [])
            or keyboard.get("layout_switches") not in (None, [])
        ):
            raise UISceneError("不可见键盘不能包含键位或切换控件。")

        preedit_bounds: list[tuple[float, float, float, float]] = []
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
            for candidate in raw_candidates:
                if not isinstance(candidate, dict) or set(candidate) != {
                    "text", "bounds", "confidence", "fully_visible"
                }:
                    raise UISceneError("IME候选字段不符合协议。")
                candidate_text = str(candidate.get("text") or "").strip()
                if not candidate_text or len(candidate_text) > 20:
                    raise UISceneError("IME候选文字格式无效。")
                if not _valid_1000_bounds(candidate.get("bounds")):
                    raise UISceneError("IME候选 bounds 无效。")
                candidate_bounds = tuple(float(value) for value in candidate["bounds"])
                candidate_confidence = _audit_confidence(
                    candidate.get("confidence"), "IME候选"
                )
                if not isinstance(candidate.get("fully_visible"), bool):
                    raise UISceneError("IME候选 fully_visible 必须是布尔值。")
                if not _bounds_inside(candidate_bounds, bounds, tolerance=12):
                    raise UISceneError("IME候选必须完整位于对应预编辑区内。")
                if candidate["fully_visible"] and candidate_confidence >= 0.9:
                    candidates.append(
                        {
                            "text": candidate_text,
                            "bounds": candidate_bounds,
                            "confidence": candidate_confidence,
                        }
                    )
            if confidence >= 0.9:
                preedit_bounds.append(bounds)
                trusted_preedits.append(
                    {
                        "text": str(item.get("text") or "").strip(),
                        "bounds": bounds,
                        "confidence": confidence,
                        "candidates": candidates,
                    }
                )

        matches: list[dict[str, Any]] = []
        for item in application_inputs:
            if not isinstance(item, dict) or set(item) != {
                "structure_id",
                "bounds",
                "fully_visible",
                "text",
                "placeholder",
                "visible_editable_cues",
                "confidence",
                "right_button",
            }:
                raise UISceneError("应用输入结构字段不符合协议。")
            button = item.get("right_button")
            if _can_discard_incomplete_separate_right_button(item, button):
                # The incomplete adjacent control grants no authority and is
                # spatially disjoint from the application input. Discard only
                # that optional object; never widen or move the input bounds.
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
            if not item["fully_visible"] or confidence < 0.9:
                continue
            text = str(item.get("text") or "").strip()
            placeholder = str(item.get("placeholder") or "").strip()
            if not text and not placeholder and not cues:
                continue
            bounds = tuple(float(value) for value in item["bounds"])
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            if (
                bounds[1] <= 10
                or bounds[3] >= 990
                or width < 240
                or not 20 <= height <= 180
                or (
                    keyboard_bounds is not None
                    and _bounds_overlap_ratio(bounds, keyboard_bounds) >= 0.25
                )
                or any(
                    _bounds_overlap_ratio(bounds, preedit) >= 0.35
                    or _bounds_overlap_ratio(preedit, bounds) >= 0.35
                    for preedit in preedit_bounds
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
                    "input_bounds": input_bounds,
                    "right_button": button_match,
                    "confidence": min(
                        confidence,
                        float(button_match["confidence"])
                        if button_match is not None
                        else confidence,
                    ),
                }
            )

        switch_is_goal = _goal_requests_keyboard_mode_switch(goal_context)
        trusted_input = matches[0] if len(matches) == 1 else None
        exact_ime_candidate: dict[str, Any] | None = None
        input_step = None
        if trusted_input is not None and _goal_has_explicit_input_text(goal_context):
            focused_context = _active_subgoal_visual_context(goal_context)
            entities = (
                focused_context.get("goal_entities")
                if focused_context is not goal_context
                else goal_context.get("entities")
            )
            target_text = entities.get("input_text") if isinstance(entities, dict) else None
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
                raw_qwerty_anchors,
                keyboard_bounds=keyboard_bounds,
            )
        generic_backspace_geometry = _validated_keyboard_backspace_key(
            keyboard.get("backspace_key"),
            keyboard_bounds=keyboard_bounds,
        )
        raw_mode_switch = keyboard.get("mode_switch")
        input_needs_mode_switch = bool(
            input_step is not None
            and input_step.kind in {"direct_latin", "chinese_pinyin"}
            and keyboard_visible
            and keyboard_layout == "qwerty"
            and keyboard_input_mode in {"direct_latin", "chinese_pinyin"}
            and keyboard_input_mode != input_step.required_mode
        )
        switch_is_goal = switch_is_goal or input_needs_mode_switch
        if (
            not switch_is_goal
            and trusted_input is not None
            and keyboard_visible
            and keyboard_layout == "qwerty"
            and keyboard_input_mode in {"direct_latin", "chinese_pinyin"}
            and _is_incomplete_optional_keyboard_mode_switch(raw_mode_switch)
        ):
            # A text-entry target does not consume the keyboard switch. When
            # the current input and a known QWERTY keyboard state are independently
            # proven, discard only an incomplete subset of the optional switch
            # schema. This applies both before input (empty value) and while
            # verifying the exact non-empty result. Action authorization still
            # requires an empty value plus direct_latin in
            # UniversalActionController, so discarding this unused optional object
            # cannot authorize typing in chinese_pinyin. Extra fields, malformed
            # geometry and all switch goals reach strict validation below and fail
            # closed.
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
        layout_switches = _validated_keyboard_layout_switches(
            keyboard.get("layout_switches", []),
            keyboard_bounds=keyboard_bounds,
            current_layout=keyboard_layout,
        )
        case_switch = _validated_keyboard_case_switch(
            keyboard.get("case_switch"),
            keyboard_bounds=keyboard_bounds,
            keyboard_layout=keyboard_layout,
            keyboard_input_mode=keyboard_input_mode,
            case_mode=keyboard_case_mode,
        )
        exact_literal_key: dict[str, Any] | None = None
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
        ):
            raise UISceneError(
                "文字输入授权要求本轮输入结构审计提供有效 QWERTY anchors。"
            )
        if trusted_input is None and (mode_switch is None or not switch_is_goal):
            return scene

        value = scene.to_dict()
        elements: list[dict[str, Any]] = []
        for element in value.get("elements") or []:
            if not isinstance(element, dict):
                continue
            element = dict(element)
            element["states"] = dict(element.get("states") or {})
            element["states"]["goal_relevant"] = False
            # A trusted audit input supersedes preliminary input proposals. Keeping
            # both would leave two overlapping high-confidence action targets and
            # correctly make unique_trusted_goal_element reject the scene.
            if trusted_input is not None and element.get("role") == "input":
                continue
            elements.append(element)
        if trusted_input is not None:
            pending_auxiliary_input_action = any(
                item is not None
                for item in (
                    exact_ime_candidate,
                    exact_literal_key,
                    exact_layout_switch,
                    exact_case_switch,
                )
            )
            input_requires_auxiliary_action = bool(
                input_step is not None
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
                    and not pending_auxiliary_input_action
                    and not input_requires_auxiliary_action
                ),
                "fully_visible": True,
                "value": trusted_input["text"],
            }
            if not keyboard_visible:
                # Absence is useful task evidence only when the dedicated
                # full-frame input audit explicitly reports keyboard.visible=false.
                # An empty preliminary overlays list alone never mints this fact.
                states["soft_keyboard_visible"] = False
            if trusted_input["placeholder"]:
                states["placeholder"] = trusted_input["placeholder"]
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
                        "ime_preedit_text": input_step.pinyin,
                        "ime_exact_candidate_text": input_step.segment,
                    }
                )
            input_label = trusted_input["text"] or trusted_input["placeholder"]
            input_evidence = list(trusted_input["visible_editable_cues"])
            if trusted_input["text"]:
                input_evidence.insert(0, f"应用输入框当前文字：{trusted_input['text']}")
            elif trusted_input["placeholder"]:
                input_evidence.insert(0, f"应用输入框为空，占位提示：{trusted_input['placeholder']}")
            if not keyboard_visible:
                input_evidence.append(AUDITED_SOFT_KEYBOARD_HIDDEN_EVIDENCE)
            elements.append(
                {
                    "element_id": "local_audited_input_1",
                    "role": "input",
                    "meaning": "application_text_input",
                    "label": input_label,
                    "bounds": [part / 1000.0 for part in trusted_input["input_bounds"]],
                    "confidence": trusted_input["confidence"],
                    "states": states,
                    "evidence": input_evidence[:6],
                }
            )
            right_button = trusted_input["right_button"]
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
                        "pinyin": input_step.pinyin,
                    },
                    "evidence": [
                        f"输入结构审计确认拼音 {input_step.pinyin} 的唯一逐字候选：{input_step.segment}"
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
        value["elements"] = elements
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


def _validated_qwerty_keyboard_geometry(
    value: Any,
    *,
    keyboard_bounds: tuple[float, float, float, float],
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
        if not (
            keyboard_bounds[0] - 20 <= x <= keyboard_bounds[2] + 20
            and keyboard_bounds[1] - 20 <= y <= keyboard_bounds[3] + 20
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
    label_mode = _keyboard_mode_implied_by_label(label)
    if label_mode is not None and label_mode != current_mode:
        raise UISceneError("mode_switch label 与 current_mode 冲突。")
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
        raise UISceneError("backspace_key 缺少逐字可见退格图形。")
    if value.get("fully_visible") is not True or not _valid_1000_bounds(
        value.get("bounds")
    ):
        raise UISceneError("backspace_key 必须完整可见且 bounds 有效。")
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
        source = item.get("current_layout")
        target = item.get("target_layout")
        if (
            source not in layouts
            or target not in layouts
            or source == target
            or source != current_layout
            or not _valid_1000_bounds(item.get("bounds"))
        ):
            raise UISceneError("layout_switch 方向或 bounds 无效。")
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


def _can_discard_incomplete_separate_right_button(
    input_item: Any,
    value: Any,
) -> bool:
    """Discard only a schema-subset control proven outside the input bounds."""

    required = {"label", "bounds", "confidence"}
    if (
        not isinstance(input_item, dict)
        or not isinstance(value, dict)
        or not set(value) < required
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


def _keyboard_mode_implied_by_label(label: str) -> str | None:
    visible = re.sub(r"[\s_\-/]+", "", str(label or "").strip().casefold())
    if visible in {"中", "中文", "chinese", "pinyin"}:
        return "chinese_pinyin"
    if visible in {"英", "en", "eng", "english", "abc", "latin"}:
        return "direct_latin"
    return None


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


def _needs_targeted_refinement(scene: UIScene, context: dict[str, Any]) -> bool:
    if not context:
        return False
    focused = _observation_goal_context(context)
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
