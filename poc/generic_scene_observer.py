from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import replace
from typing import Any, Iterator

from PIL import Image

from observation_images import (
    VisualObstruction,
    consensus_top_edge_obstructions,
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
from ui_scene import (
    ALLOWED_ROLES,
    CameraAlignmentFacts,
    MIN_TARGET_CONFIDENCE,
    SystemUIFacts,
    UI_SCENE_PROTOCOL_VERSION,
    UIScene,
    UISceneError,
)
from vision_agent import VisionAgentError, _extract_json_object, _image_data_url


GENERIC_SCENE_OBSERVER_VERSION = "2026-08-15-generic-scene-observer-v17"
INPUT_STRUCTURE_AUDIT_VERSION = "2026-08-14-input-structure-audit-v2"
SYSTEM_UI_AUDIT_VERSION = "2026-08-14-system-ui-audit-v1"
COMPACT_OUTPUT_TOKENS = 1200
TARGETED_OUTPUT_TOKENS = 1200
INPUT_STRUCTURE_AUDIT_TOKENS = 700
SYSTEM_UI_AUDIT_TOKENS = 600
ORIENTATION_AUDIT_TOKENS = 500
MIN_SYSTEM_UI_AUDIT_CONFIDENCE = 0.80
OBSERVATION_TIMEOUT_SECONDS = 60.0
MAX_COMPACT_ELEMENTS = 12

STAGE_LABELS = {
    "idle": "空闲",
    "checking_stability": "检查画面稳定性",
    "waiting_compact_observation": "等待千问快速观察",
    "parsing_compact_observation": "解析快速观察结果",
    "waiting_compact_retry": "等待千问修正观察格式",
    "parsing_compact_retry": "解析修正结果",
    "waiting_targeted_refinement": "等待千问目标精查",
    "parsing_targeted_refinement": "解析目标精查结果",
    "waiting_input_structure_audit": "等待输入结构只读审计",
    "parsing_input_structure_audit": "解析输入结构只读审计",
    "waiting_system_ui_audit": "等待系统界面只读审计",
    "parsing_system_ui_audit": "解析系统界面只读审计",
    "waiting_system_ui_audit_retry": "等待系统界面审计格式修正",
    "waiting_orientation_audit": "等待独立方向只读审计",
    "parsing_orientation_audit": "解析独立方向只读审计",
    "completed": "观察完成",
    "failed": "观察安全停止",
}


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
        try:
            raw = self._provider_chat(
                [_json_only_system_message(), {"role": "user", "content": content}],
                max_tokens=ORIENTATION_AUDIT_TOKENS,
            )
            self._set_stage("parsing_orientation_audit")
            payload = _parse_orientation_audit(raw)
            credential = _mint_audited_credential(
                device_id=device_id,
                scene_fingerprint=scene_fingerprint,
                frame=frame,
                phone_content_rotation=payload["phone_content_rotation"],
                confidence=payload["confidence"],
                evidence=tuple(payload["evidence"]),
            )
            credential.assert_authorizes(
                device_id=device_id,
                scene_fingerprint=scene_fingerprint,
                frame_size=tuple(frame.size),
            )
            self.last_orientation_audit_diagnostics = {
                "audit_version": ORIENTATION_AUDIT_PROTOCOL_VERSION,
                "model_calls": 1,
                "image_count": 3,
                "cache_hit": False,
                "selected_frame_index": selected_index,
                "frame_size": list(frame.size),
                "frame_fingerprint": local_fingerprint,
                "confidence": float(credential.confidence),
                "response_payload": _orientation_audit_diagnostic_payload(raw),
                "audit_accepted": True,
            }
            return credential
        except Exception as exc:
            self.last_orientation_audit_diagnostics = {
                "audit_version": ORIENTATION_AUDIT_PROTOCOL_VERSION,
                "model_calls": 1,
                "image_count": 3,
                "cache_hit": False,
                "selected_frame_index": selected_index,
                "frame_size": list(frame.size),
                "frame_fingerprint": local_fingerprint,
                "response_payload": _orientation_audit_diagnostic_payload(raw),
                "audit_accepted": False,
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
                "observation_timeout_seconds": OBSERVATION_TIMEOUT_SECONDS,
                "max_compact_elements": MAX_COMPACT_ELEMENTS,
                "last_scene_enum_values": dict(
                    self.last_diagnostics.get("scene_enum_values") or {}
                ),
                "last_orientation_audit_diagnostics": dict(
                    self.last_orientation_audit_diagnostics
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
        self.last_diagnostics = {}
        self._set_stage("checking_stability")
        started = time.perf_counter()
        model_calls = 0
        compact_retry_used = False
        format_retry_used = False
        local_structural_repair_used = False
        targeted_refinement_used = False
        input_structure_audit_used = False
        system_ui_audit_used = False
        system_ui_audit_retry_used = False
        system_ui_audit_confidence: float | None = None
        system_ui_audit_evidence: tuple[str, ...] = ()
        targeted_roi_bounds: tuple[int, int, int, int] | None = None
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
                payload = _extract_compact_json_object(value)
                return _suppress_obscured_input_evidence(
                    _parse_scene(
                        json.dumps(
                            payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        fingerprint=fingerprint,
                        goal_context=context,
                        allow_invalid_system_ui_unknown=system_ui_audit_required,
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
                    allow_invalid_system_ui_unknown=system_ui_audit_required,
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
                and _needs_targeted_refinement(scene, context)
            ):
                targeted_refinement_used = True
                targeted_roi_bounds = _goal_directed_roi_bounds(context)
                detail_image_part = image_part
                if targeted_roi_bounds is not None:
                    detail_frame = _crop_normalized(frame, targeted_roi_bounds)
                    detail_image_part = {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(detail_frame)},
                    }
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
                                    roi_bounds=targeted_roi_bounds,
                                ),
                                },
                                image_part,
                            ]
                            + ([detail_image_part] if targeted_roi_bounds is not None else [])
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
                        _parse_scene(
                            raw,
                            fingerprint=fingerprint,
                            goal_context=context,
                            allow_invalid_system_ui_unknown=False,
                            camera_alignment_override=scene.camera_alignment,
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
                    ):
                        raise
                    format_retry_used = True
                    self._set_stage("waiting_compact_retry")
                    targeted_retry_messages = [
                        _json_only_system_message(),
                        {
                            "role": "user",
                            "content": (
                                [
                                    {
                                    "type": "text",
                                    "text": _targeted_retry_prompt(
                                        context,
                                        targeted_error,
                                        roi_bounds=targeted_roi_bounds,
                                    ),
                                    },
                                    image_part,
                                ]
                                + ([detail_image_part] if targeted_roi_bounds is not None else [])
                            ),
                        }
                    ]
                    raw = model_chat(
                        targeted_retry_messages,
                        max_tokens=TARGETED_OUTPUT_TOKENS,
                    )
                    self.last_raw_response = raw
                    self._set_stage("parsing_compact_retry")
                    scene = _suppress_obscured_input_evidence(
                        _parse_scene(
                            raw,
                            fingerprint=fingerprint,
                            goal_context=context,
                            allow_invalid_system_ui_unknown=False,
                            camera_alignment_override=scene.camera_alignment,
                        ),
                        visual_obstructions,
                        fingerprint=fingerprint,
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
                try:
                    scene, system_ui_audit_confidence, system_ui_audit_evidence = (
                        _apply_system_ui_audit(
                            scene,
                            raw,
                            fingerprint=fingerprint,
                        )
                    )
                except VisionAgentError as audit_error:
                    system_ui_audit_retry_used = True
                    self._set_stage("waiting_system_ui_audit_retry")
                    retry_messages = [
                        _json_only_system_message(),
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": _system_ui_audit_retry_prompt(
                                        context,
                                        audit_error,
                                    ),
                                },
                                *system_ui_images,
                            ],
                        },
                    ]
                    raw = model_chat(
                        retry_messages,
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
                self._set_stage("waiting_input_structure_audit")
                audit_content: list[dict[str, Any]] = [
                    {
                        "type": "text",
                        "text": _input_structure_audit_prompt(
                            context,
                            roi_bounds=targeted_roi_bounds,
                        ),
                    },
                    image_part,
                ]
                if targeted_roi_bounds is not None:
                    audit_content.append(detail_image_part)
                raw = model_chat(
                    [
                        _json_only_system_message(),
                        {"role": "user", "content": audit_content},
                    ],
                    max_tokens=INPUT_STRUCTURE_AUDIT_TOKENS,
                )
                self.last_raw_response = raw
                self._set_stage("parsing_input_structure_audit")
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

            target_local_candidate = scene.unique_trusted_goal_element()
            completion_evidence = scene.trusted_completion_evidence()
            if not scene.stable or (
                float(scene.confidence) < MIN_TARGET_CONFIDENCE
                and target_local_candidate is None
                and not completion_evidence
            ):
                self.last_diagnostics = {
                    "observer_version": GENERIC_SCENE_OBSERVER_VERSION,
                    "strategy": "compact_then_targeted_on_demand",
                    "model_calls": model_calls,
                    "compact_retry_used": compact_retry_used,
                    "format_retry_used": format_retry_used,
                    "local_structural_repair_used": local_structural_repair_used,
                    "targeted_refinement_used": targeted_refinement_used,
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
                "strategy": "compact_then_targeted_on_demand",
                "model_calls": model_calls,
                "compact_retry_used": compact_retry_used,
                "format_retry_used": format_retry_used,
                "local_structural_repair_used": local_structural_repair_used,
                "first_pass_success": not format_retry_used,
                "repair_retry_success": format_retry_used,
                "targeted_refinement_used": targeted_refinement_used,
                "input_structure_audit_used": input_structure_audit_used,
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
                    "model_calls": model_calls,
                    "compact_retry_used": compact_retry_used,
                    "format_retry_used": format_retry_used,
                    "local_structural_repair_used": local_structural_repair_used,
                    "first_pass_success": False,
                    "repair_retry_success": False,
                    "targeted_refinement_used": targeted_refinement_used,
                    "input_structure_audit_used": input_structure_audit_used,
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
            self.last_diagnostics = base
            raise
        finally:
            self._set_stage("idle")

    def _provider_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
    ) -> str:
        try:
            return self.provider._chat(
                messages,
                max_tokens=max_tokens,
                timeout=OBSERVATION_TIMEOUT_SECONDS,
                max_attempts=2,
            )
        except TypeError as exc:
            # Keep simple test providers and local replay providers compatible.
            # Production DashScopeVisionProvider accepts the explicit limits.
            text = str(exc)
            if "unexpected keyword" not in text and "keyword argument" not in text:
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

INPUT_VALUE_OBSERVATION_RULE = (
    "role=input且框内文字清晰可读时，必须在states.value中逐字填写当前可见文字；空框写空字符串，"
    "看不清才省略value，禁止根据目标补写。软键盘可见时还必须在states.keyboard_layout写"
    "qwerty、numeric、symbol或unknown，并在states.keyboard_input_mode写direct_latin、"
    "chinese_pinyin或unknown。QWERTY只描述按键排列，绝不等于英文直输：画面出现中文候选、"
    "拼音分词撇号或明确中文模式时必须写chinese_pinyin；只有明确显示英文/Latin直输模式时才能写"
    "direct_latin；看不清写unknown。这些都只是画面事实，不授权输入。若键盘底部清楚可见独立的"
    "中/英模式切换键，必须另建role=button元素，meaning写switch_keyboard_input_mode，label逐字抄"
    "可见键面文字，states写keyboard_input_mode_switch:true、current_mode和target_mode；不确定当前"
    "模式或切换方向时不得编造该元素。字母、数字、退格、回车等普通键仍必须role=keyboard_key。"
    "若非空输入框内部或紧邻右侧清楚可见独立的圆形×/清空图标，必须另建role=button或icon元素，"
    "meaning写clear_local_text，states写local_text_clear:true，label必须逐字写图标本身的×/✕/✖/x；"
    "若看不清真实叉号图形或只能自由描述为叉号，就不得标记local_text_clear。只框该图标自身，不能与输入框合并，"
    "也绝不能把键盘退格键/删除键标成local_text_clear。页面右侧的文字‘取消’/cancel是取消编辑或"
    "退出控件，不是本地清空图标；必须meaning=cancel且goal_relevant:false，绝不能标成clear_local_text。"
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
structure establishes axes. Do not return coordinates, bounds, actions or plans.
Return exactly this JSON object and no Markdown:
{{"protocol_version":"{ORIENTATION_AUDIT_PROTOCOL_VERSION}",
"phone_content_rotation":"upright|rotated_90|rotated_180|rotated_270|unknown",
"confidence":0.0,"evidence":["one or two short phone-only facts"]}}
""".strip()


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


def _compact_prompt(context: dict[str, Any]) -> str:
    return f"""
你是通用手机页面观察器，只报告画面事实，不规划也不执行动作。
用户目标只用于选择需要读清的控件，不能让你幻读：
{json.dumps(context, ensure_ascii=False, separators=(',', ':'))}

用最短JSON报告：当前前台App、页面类型、最上层弹层，以及与目标直接相关的可见控件。
规则：
1. 桌面写 launcher；不确定写 unknown。不得把目标App当成当前App。
2. elements最多{MAX_COMPACT_ELEMENTS}个，只保留目标相关控件、关闭/返回、当前输入框和必要导航。
3. bounds使用0..1000的[left,top,right,bottom]，必须只框真实清晰控件。
4. role仅限button/icon/input/text/tab/toggle/image/list_item/dialog/keyboard_key/container/unknown。
5. meaning用lower_snake_case。与目标直接相关的控件在states中写goal_relevant:true。
6. evidence只抄画面短文字或明确外观。看不清就降低confidence或省略元素。
7. 禁止action、plan、step、tap、swipe、command、coordinates等动作字段。
   overlays只允许简短字符串名称；任何带边界、角色或ID的可交互候选必须放入elements，
   不得把对象放入overlays。
8. 场景confidence只评价当前画面本身是否清楚、稳定、可描述，不评价目标是否已完成或目标控件
   是否存在。清晰稳定的页面即使没有目标控件，也应保持与画面质量一致的高confidence并返回空
   elements；只有模糊、遮挡、过渡或无法判断页面事实时才降低confidence。
9. {PREFILLED_INPUT_OBSERVATION_RULE}
10. {INPUT_VALUE_OBSERVATION_RULE}
11. {SYSTEM_UI_OBSERVATION_RULE}
12. {CAMERA_ALIGNMENT_OBSERVATION_RULE}

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


def _targeted_retry_prompt(
    context: dict[str, Any],
    error: Exception,
    *,
    roi_bounds: tuple[int, int, int, int] | None = None,
) -> str:
    return f"""
上一次目标精查输出不是完整、合法的页面观察JSON，控制器没有产生任何候选动作。
错误摘要：{str(error)[:300]}
目标上下文：{json.dumps(context, ensure_ascii=False, separators=(',', ':'))}
{_roi_observation_note(roi_bounds)}
这是本轮观察唯一一次格式修复。请重新独立观察原图，只返回最小完整JSON；没有可靠目标就返回空elements并降低confidence。
格式修复不能靠删除真实候选通过；原图中清楚可见且与目标直接相关的入口必须改写为elements，
即使目标最终结果尚未出现。只有重新观察后仍无法确认时才返回空elements。
格式：
{{"protocol_version":"{UI_SCENE_PROTOCOL_VERSION}","foreground_app_id":"unknown",
"screen_id":"unknown","summary":"短描述","system_ui":{{"immersive_or_fullscreen":"unknown",
"navigation_bar_visible":"unknown"}},"elements":[],"overlays":[],
"stable":true,"confidence":0.0,"fingerprint":""}}
元素仅允许element_id、role、meaning、label、bounds、confidence、states、evidence；禁止动作、计划和裸坐标。不要Markdown。
bounds必须是恰好4个0..1000数值的数组[left,top,right,bottom]；不能是x/y/width/height对象、两个点或嵌套数组。
overlays只能是字符串数组；可交互候选必须放入elements并使用element_id，不能把对象放入overlays。
{SYSTEM_UI_OBSERVATION_RULE}
输入框识别规则：{PREFILLED_INPUT_OBSERVATION_RULE}
输入框文字与键盘规则：{INPUT_VALUE_OBSERVATION_RULE}
"""


def _targeted_prompt(
    context: dict[str, Any],
    *,
    first_scene: dict[str, Any],
    roi_bounds: tuple[int, int, int, int] | None = None,
) -> str:
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
只保留最多4个最相关元素；目标元素必须states.goal_relevant=true。看不清或不唯一就不要输出，
并降低场景confidence。坐标0..1000，只框元素自身。禁止任何动作、计划或建议字段。
目标相关元素既包括已经满足完成条件的可见结果，也包括画面上清楚可见、能使该结果进入视野
的入口控件；这里只报告控件事实，不建议也不授权使用它。
置信度只评价当前画面观察本身是否可靠，不能因为目标尚未完成而降低；例如清晰桌面上唯一目标
应用入口可形成高可信观察，即使应用尚未打开。模糊、遮挡或不唯一时仍必须降低，禁止虚增。
目标相关控件确实不存在时返回空elements，但只要页面事实清楚稳定，场景confidence仍应保持高值；
不得因为系统级动作没有屏内按钮、或因为未找到目标控件，就把清晰页面写成低置信。
    输入框识别规则：{PREFILLED_INPUT_OBSERVATION_RULE}
    输入框文字与键盘规则：{INPUT_VALUE_OBSERVATION_RULE}
如果能清楚看见相关横向边框、框内文字和右侧独立搜索/提交按钮，但仍无法判断边框是否可编辑，
不得因此返回空elements：请分别报告container、其内部text和右侧button的真实边界与证据；
这三个元素都必须在states中明确写fully_visible:true或false。若画面边缘还有被裁切的相似结构，
只能在summary说明，不能把它标成目标；优先报告四边完整可见的结构。完整container和text写
goal_relevant:true，相邻button写goal_relevant:false。本地只会在三者都fully_visible:true且严格
几何关系成立时把这组只读事实归一化，绝不会因此激活按钮。
只返回完整JSON：
{{"protocol_version":"{UI_SCENE_PROTOCOL_VERSION}","foreground_app_id":"unknown",
"screen_id":"unknown","summary":"目标精查后的当前画面","system_ui":{{"immersive_or_fullscreen":"unknown",
"navigation_bar_visible":"unknown"}},"elements":[],"overlays":[],
"stable":true,"confidence":0.0,"fingerprint":""}}
元素仅允许element_id、role、meaning、label、bounds、confidence、states、evidence。不要Markdown。
role仅限button/icon/input/text/tab/toggle/image/list_item/dialog/keyboard_key/container/unknown。
container仅表示与目标有关的页面内容区域；tab_group、tab_bar和toolbar等其他非点击结构只写进summary。
{SYSTEM_UI_OBSERVATION_RULE}
overlays只能是字符串数组；任何可交互候选都必须放入elements并使用element_id，
不得把带bounds、role或ID的对象放入overlays。
"""


def _input_structure_audit_prompt(
    context: dict[str, Any],
    *,
    roi_bounds: tuple[int, int, int, int] | None,
) -> str:
    return f"""
You are a read-only, app-independent UI structure auditor. The normal scene observer did not establish an input target.
Goal context (evidence selection only): {json.dumps(context, ensure_ascii=False, separators=(',', ':'))}
Image 1 is always the complete phone frame. {_input_audit_detail_note(roi_bounds)}
Distinguish three different visual structures; never merge them:
1. application_inputs: editable search/address/form fields in the App content area. Include an empty field only when a complete border plus a visible placeholder, caret, focus highlight, or other literal editable cue is visible.
2. ime_preedit_regions: the input method's composition/candidate strip. It is never an application input, even when it contains composed text and a trailing icon.
3. keyboard.mode_switch: one compact key inside the visible keyboard that explicitly switches between chinese_pinyin and direct_latin. Ordinary letters, backspace, enter, robot/assistant, voice, emoji, and candidate-strip icons are never mode switches.
Do not plan, suggest, authorize, or perform any action. All bounds MUST use Image 1 full-frame normalized coordinates 0..1000.
Use text="" for a visibly empty application field. Copy placeholders and visible_editable_cues literally; do not infer them from the goal. right_button describes a trailing utility control; it is structural evidence only and is never authorized for activation. Set it to null when no separate trailing control is visible.
Return exactly this JSON schema and no other fields:
{{"protocol_version":"{INPUT_STRUCTURE_AUDIT_VERSION}",
"application_inputs":[{{"structure_id":"app-input-1","bounds":[0,0,1000,1000],
"fully_visible":true,"text":"","placeholder":"visible placeholder or empty",
"visible_editable_cues":["literal visible cue"],"confidence":0.0,
"right_button":null}}],
"ime_preedit_regions":[{{"region_id":"ime-preedit-1","bounds":[0,0,1000,1000],
"text":"visible composition text or empty","confidence":0.0}}],
"keyboard":{{"visible":true,"bounds":[0,0,1000,1000],"layout":"qwerty",
"input_mode":"chinese_pinyin","mode_switch":{{"label":"中","bounds":[0,0,1000,1000],
"confidence":0.0,"current_mode":"chinese_pinyin","target_mode":"direct_latin"}}}}}}
When no keyboard is visible, keyboard must be {{"visible":false,"bounds":null,"layout":"unknown","input_mode":"unknown","mode_switch":null}}.
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
) -> UIScene | None:
    """Accept one punctuation edit only when exactly one strict scene survives."""

    result = _unique_strict_structural_scene_edit(
        raw,
        fingerprint=fingerprint,
        goal_context=goal_context,
        allow_invalid_system_ui_unknown=allow_invalid_system_ui_unknown,
        camera_layout_orientation=camera_layout_orientation,
    )
    return result[1] if result is not None else None


def _unique_strict_structural_scene_edit(
    raw: str,
    *,
    fingerprint: str,
    goal_context: dict[str, Any] | None = None,
    allow_invalid_system_ui_unknown: bool = False,
    camera_layout_orientation: str | None = None,
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
        _normalize_prefilled_input_structure(payload, goal_context or {})
        _normalize_local_text_clear_structure(payload, goal_context or {})
        _normalize_unique_input_focus(payload)
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
    """Select at most one coarse ROI from explicit spatial words in the goal."""

    visible = json.dumps(context, ensure_ascii=False).casefold()
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


def _goal_requests_input(context: dict[str, Any]) -> bool:
    visible = json.dumps(context, ensure_ascii=False).casefold()
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
            "input field",
            "search box",
            "text field",
            "editable field",
            "address bar",
            "textbox",
            "input_text",
            "输入模式",
            "直输模式",
            "键盘模式",
            "input mode",
            "keyboard mode",
            "direct_latin",
            "chinese_pinyin",
        )
    )


def _goal_requests_system_ui_audit(context: dict[str, Any]) -> bool:
    visible = json.dumps(context, ensure_ascii=False).casefold()
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


def _goal_requests_keyboard_mode_switch(context: dict[str, Any]) -> bool:
    visible = json.dumps(context, ensure_ascii=False).casefold()
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
    keyboard_is_relevant = states.get("focused") is True or _scene_reports_keyboard(
        scene
    )
    if not keyboard_is_relevant:
        return False
    return not (
        states.get("fully_visible") is True
        and states.get("focused") is True
        and states.get("keyboard_layout") in {"qwerty", "numeric", "symbol"}
        and states.get("keyboard_input_mode")
        in {"direct_latin", "chinese_pinyin"}
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
        if not isinstance(keyboard, dict) or set(keyboard) != {
            "visible",
            "bounds",
            "layout",
            "input_mode",
            "mode_switch",
        }:
            raise UISceneError("输入结构审计 keyboard 字段不符合协议。")

        keyboard_visible = keyboard.get("visible")
        keyboard_layout = keyboard.get("layout")
        keyboard_input_mode = keyboard.get("input_mode")
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
        keyboard_bounds: tuple[float, float, float, float] | None = None
        if keyboard_visible:
            if not _valid_1000_bounds(keyboard.get("bounds")):
                raise UISceneError("可见键盘必须提供有效 bounds。")
            keyboard_bounds = tuple(float(value) for value in keyboard["bounds"])
            if (
                keyboard_bounds[2] - keyboard_bounds[0] < 300
                or keyboard_bounds[3] - keyboard_bounds[1] < 180
            ):
                raise UISceneError("可见键盘 bounds 过小，不能建立键盘区域。")
        elif keyboard.get("bounds") is not None or keyboard.get("mode_switch") is not None:
            raise UISceneError("不可见键盘不能包含 bounds 或 mode_switch。")

        preedit_bounds: list[tuple[float, float, float, float]] = []
        for item in ime_preedit_regions:
            if not isinstance(item, dict) or set(item) != {
                "region_id",
                "bounds",
                "text",
                "confidence",
            }:
                raise UISceneError("IME预编辑区字段不符合协议。")
            if not _valid_1000_bounds(item.get("bounds")):
                raise UISceneError("IME预编辑区 bounds 不符合0..1000协议。")
            confidence = _audit_confidence(item.get("confidence"), "IME预编辑区")
            bounds = tuple(float(value) for value in item["bounds"])
            if confidence >= 0.9:
                preedit_bounds.append(bounds)

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

        mode_switch = _validated_keyboard_mode_switch(
            keyboard.get("mode_switch"),
            keyboard_bounds=keyboard_bounds,
        )
        if (
            mode_switch is not None
            and keyboard_input_mode != "unknown"
            and mode_switch["current_mode"] != keyboard_input_mode
        ):
            raise UISceneError("模式切换键 current_mode 与键盘 input_mode 冲突。")
        switch_is_goal = _goal_requests_keyboard_mode_switch(goal_context)
        trusted_input = matches[0] if len(matches) == 1 else None
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
            states: dict[str, Any] = {
                "goal_relevant": not switch_is_goal,
                "fully_visible": True,
                "value": trusted_input["text"],
            }
            if trusted_input["placeholder"]:
                states["placeholder"] = trusted_input["placeholder"]
            if keyboard_bounds is not None:
                states.update(
                    {
                        "focused": True,
                        "keyboard_layout": keyboard_layout,
                        "keyboard_input_mode": keyboard_input_mode,
                    }
                )
            input_label = trusted_input["text"] or trusted_input["placeholder"]
            input_evidence = list(trusted_input["visible_editable_cues"])
            if trusted_input["text"]:
                input_evidence.insert(0, f"应用输入框当前文字：{trusted_input['text']}")
            elif trusted_input["placeholder"]:
                input_evidence.insert(0, f"应用输入框为空，占位提示：{trusted_input['placeholder']}")
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
                        "keyboard_input_mode_switch": True,
                        "current_mode": mode_switch["current_mode"],
                        "target_mode": mode_switch["target_mode"],
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


def _audit_confidence(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UISceneError(f"{field_name} confidence 格式无效。")
    confidence = float(value)
    if not 0.0 <= confidence <= 1.0:
        raise UISceneError(f"{field_name} confidence 超出0..1。")
    return confidence


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
    for item in elements:
        if not isinstance(item, dict):
            continue
        states = item.get("states")
        if not isinstance(states, dict):
            continue
        role = str(item.get("role") or "").strip()
        if role != "input" and states.get("goal_relevant") is not True:
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
    if scene.confidence < 0.72:
        return True
    if any(element.states.get("goal_relevant") is True for element in scene.elements):
        return False
    if scene.screen_id == "unknown":
        return True

    target_app = str(context.get("app_id") or "").strip().casefold()
    objective = str(context.get("objective") or "").strip()
    if target_app and scene.foreground_app_id.casefold() == target_app:
        if re.search(r"^(打开|进入|启动)", objective):
            return False

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
    values: list[str] = []
    for key in ("app_id", "app_name"):
        value = str(context.get(key) or "").strip().casefold()
        if value:
            values.append(value)
    entities = context.get("entities")
    if isinstance(entities, dict):
        values.extend(str(value).strip().casefold() for value in entities.values())
    objective = str(context.get("objective") or "").strip().casefold()
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
