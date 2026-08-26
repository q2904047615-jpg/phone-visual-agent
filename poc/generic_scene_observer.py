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
from dataclasses import dataclass, replace
from typing import Any, Callable

from PIL import Image
from agent.domain.canonical_action_protocol import StateExpectation
from agent.domain.canonical_action_kinds import (
    CANONICAL_ACTION_KINDS as CANONICAL_ACTIONS,
)
from agent.infrastructure.observation_images import (
    consensus_top_edge_obstructions,
    local_frame_fingerprint,
    measure_frame_sharpness,
    measure_local_stability,
)
from agent.domain.visual_evidence import VisualObstruction
from qwen_runtime_errors import (
    classify_qwen_error,
)
from robot_core import WorkflowNotReady, qwerty_keyboard_config_from_anchors
from agent.domain.ui_scene import (
    ALLOWED_ROLES,
    CAMERA_LAYOUT_ORIENTATIONS,
    CameraAlignmentFacts,
    MIN_TARGET_CONFIDENCE,
    PHONE_CONTENT_ROTATIONS,
    UI_SCENE_PROTOCOL_VERSION,
    UIScene,
    UISceneError,
    camera_alignment_evidence_is_safe,
)
from vision_agent import _extract_json_object, _image_data_url, _image_request_size
from agent.domain.vision_model import VisionAgentError, public_model_identity
from agent.domain.verified_text_transaction import (
    VerifiedTextTransactionError,
    next_keyboard_layout_towards,
    plan_next_verified_input,
    preferred_keyboard_layout,
    required_keyboard_input_mode_for_step,
)
from agent.application.input_value_lineage import (
    TypedInputLineageStorePort,
    lineage_matches_persisted_surface_cue,
    lineage_matches_trailing_newline_cue,
    lineage_matches_visual,
)
from agent.domain.input_value_lineage import (
    TypedInputLineage,
)
import agent.domain.generic_goal as generic_goal_domain

SINGLE_STEP_SCENE_OBSERVER_VERSION = "2026-08-25-single-step-scene-observer-v2"
SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION = (
    "2026-08-25-single-step-qwen-observation-v2"
)
POST_ACTION_VISUAL_CONTEXT_VERSION = (
    "2026-08-25-local-post-action-visual-context-v1"
)
POST_NAVIGATION_RESULT_OBSERVATION_PHASE = "verified_navigation_result_v1"
FUSED_POST_ACTION_NEXT_STEP_OBSERVATION_PHASE = (
    "2026-08-24-verified-previous-and-plan-next-v1"
)
POST_NAVIGATION_RESULT_OBJECTIVE = "观察本次导航后的当前稳定画面"
POST_NAVIGATION_RESULT_COMPLETION_CONDITIONS = ["当前稳定结果画面已被重新观察"]
INPUT_STRUCTURE_AUDIT_VERSION = "2026-08-25-input-structure-audit-v11"
SINGLE_STEP_OUTPUT_TOKENS = 5200
OBSERVATION_TIMEOUT_SECONDS = 60.0
MAX_COMPACT_ELEMENTS = 12
AUDITED_SOFT_KEYBOARD_HIDDEN_EVIDENCE = "输入结构只读审计确认软键盘不可见"


@dataclass(frozen=True)
class PostActionVisualContext:
    """Local, typed context for the first observation after one action.

    This object records only that one canonical action reached the physical
    transport and which typed postconditions the already-validated canonical
    transition expects.  It deliberately has no ``matched`` field: only the
    fresh pixels and the Controller may decide whether the action succeeded.
    """

    canonical_action_kind: str
    expected_postconditions: tuple[StateExpectation, ...]
    protocol_version: str = POST_ACTION_VISUAL_CONTEXT_VERSION
    execution_state: str = "physical_action_executed"
    outcome: str = "pending_visual_verification"

    def validate(self) -> None:
        if self.protocol_version != POST_ACTION_VISUAL_CONTEXT_VERSION:
            raise VisionAgentError("动作后视觉上下文协议版本无效。")
        if self.execution_state != "physical_action_executed":
            raise VisionAgentError("动作后视觉上下文没有证明物理动作已执行。")
        if self.outcome != "pending_visual_verification":
            raise VisionAgentError("动作后视觉上下文不得提前声明动作匹配结果。")
        if self.canonical_action_kind not in CANONICAL_ACTIONS:
            raise VisionAgentError("动作后视觉上下文包含非canonical动作。")
        if not 1 <= len(self.expected_postconditions) <= 16:
            raise VisionAgentError("动作后视觉上下文必须包含1..16个typed后置条件。")
        for expectation in self.expected_postconditions:
            expectation.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "protocol_version": self.protocol_version,
            "execution_state": self.execution_state,
            "outcome": self.outcome,
            "canonical_action_kind": self.canonical_action_kind,
            "expected_postconditions": [
                item.to_dict() for item in self.expected_postconditions
            ],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PostActionVisualContext":
        if not isinstance(value, dict) or set(value) != {
            "protocol_version",
            "execution_state",
            "outcome",
            "canonical_action_kind",
            "expected_postconditions",
        }:
            raise VisionAgentError("动作后视觉上下文结构无效。")
        raw_expectations = value.get("expected_postconditions")
        if not isinstance(raw_expectations, list):
            raise VisionAgentError("动作后视觉上下文的typed后置条件必须是数组。")
        expectations: list[StateExpectation] = []
        for item in raw_expectations:
            if not isinstance(item, dict):
                raise VisionAgentError("动作后视觉上下文包含无效typed后置条件。")
            required = {"subject_ref", "predicate", "operator"}
            allowed = required | {"value"}
            if not required.issubset(item) or set(item) - allowed:
                raise VisionAgentError("动作后视觉上下文包含无效typed后置条件。")
            operator = item.get("operator")
            if operator in {"equals", "not_equals"}:
                if "value" not in item:
                    raise VisionAgentError("动作后视觉上下文的等值条件缺少value。")
            elif "value" in item:
                raise VisionAgentError("动作后视觉上下文的非等值条件不得携带value。")
            expectations.append(
                StateExpectation(
                    subject_ref=str(item.get("subject_ref") or ""),
                    predicate=str(item.get("predicate") or ""),
                    operator=str(operator or ""),
                    value=item.get("value"),
                )
            )
        context = cls(
            protocol_version=str(value.get("protocol_version") or ""),
            execution_state=str(value.get("execution_state") or ""),
            outcome=str(value.get("outcome") or ""),
            canonical_action_kind=str(value.get("canonical_action_kind") or ""),
            expected_postconditions=tuple(expectations),
        )
        context.validate()
        return context

STAGE_LABELS = {
    "idle": "空闲",
    "checking_stability": "检查画面稳定性",
    "waiting_single_step_observation": "等待千问单步完整观察",
    "parsing_single_step_observation": "解析单步完整观察",
    "completed": "观察完成",
    "failed": "观察安全停止",
}








class _SingleStepObserverBase:
    """Shared state and transport for the sole production scene observer."""

    def __init__(
        self,
        provider: Any,
        *,
        input_lineage_store: TypedInputLineageStorePort | None = None,
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
        self.supports_typed_input_continuation = True
        self.supports_post_action_visual_context = True
        self._observation_cache_lock = threading.RLock()
        self._observation_cache: OrderedDict[str, UIScene] = OrderedDict()
        self._observation_cache_limit = 32


    def _set_stage(self, stage: str) -> None:
        with self._stage_lock:
            self._current_stage = stage
            if stage != "idle":
                self._last_stage = stage


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


class SingleStepGenericSceneObserver(_SingleStepObserverBase):
    """Production observer backed by at most one Qwen request per fresh scene.

    The response carries the ordinary scene and, only for an input-related
    subgoal, the complete input/IME/keyboard structure in one envelope. There
    is no multi-call fallback or alternate observation authority.
    """

    def status(self) -> dict[str, Any]:
        value = dict(self.provider.status())
        with self._stage_lock:
            current_stage = self._current_stage
            last_stage = self._last_stage
        value.update(
            {
                "observer_version": SINGLE_STEP_SCENE_OBSERVER_VERSION,
                "scene_protocol": UI_SCENE_PROTOCOL_VERSION,
                "single_step_observation_protocol": (
                    SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION
                ),
                "model_role": "single_step_fused_observation",
                "supported_app_scope": "dynamic",
                "hardware_actions_enabled": False,
                "current_stage": current_stage,
                "current_stage_label": STAGE_LABELS.get(
                    current_stage,
                    current_stage,
                ),
                "last_stage": last_stage,
                "last_stage_label": STAGE_LABELS.get(last_stage, last_stage),
                "max_online_calls_per_observation": 1,
                "single_step_output_tokens": SINGLE_STEP_OUTPUT_TOKENS,
                "observation_timeout_seconds": OBSERVATION_TIMEOUT_SECONDS,
                "last_scene_enum_values": dict(
                    self.last_diagnostics.get("scene_enum_values") or {}
                ),
                "last_input_structure_shape": dict(
                    self.last_diagnostics.get("input_structure_shape") or {}
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
        post_action_context: PostActionVisualContext | dict[str, Any] | None = None,
    ) -> UIScene:
        # Never reuse model-authored facts from the prior scene.  The only
        # cross-step visual context is a locally minted typed action summary;
        # current pixels remain the sole source of current-screen facts.
        del prior_scene
        if isinstance(post_action_context, dict):
            post_action_context = PostActionVisualContext.from_dict(
                post_action_context
            )
        elif post_action_context is not None:
            post_action_context.validate()
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
            fingerprint = local_frame_fingerprint(frame)
            context = generic_goal_domain.safe_goal_context(goal_context or {})
            cache_key = _observation_cache_key(
                device_id=device_id,
                fingerprint=fingerprint,
                goal_context=context,
                input_lineage=input_lineage_override,
                post_action_context=post_action_context,
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
                        "post_action_visual_context": (
                            post_action_context.to_dict()
                            if post_action_context is not None
                            else None
                        ),
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
            model_frames = (
                tuple(frames[stable_tail_start:])
                if input_structure_required
                else (frame,)
            )

            request_image_sizes = {
                _image_request_size(item) for item in model_frames
            }
            if len(request_image_sizes) != 1:
                raise VisionAgentError(
                    "同一步发送给Qwen的稳定帧尺寸不一致，不能建立唯一坐标空间。"
                )
            request_image_size = next(iter(request_image_sizes))

            prompt = _single_step_observation_prompt(
                context,
                include_input_structure=input_structure_required,
                current_input_text=ledger_value_hint,
                image_count=len(model_frames),
                request_image_size=request_image_size,
                post_action_context=post_action_context,
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
                request_image_size=request_image_size,
            )
            scene_payload = dict(envelope["scene"])
            if not input_structure_required:
                # The retired multi-call observer already treated an
                # out-of-range goal-marked batch as unusable geometry while
                # preserving the independently typed top-level App/screen
                # observation.  Carry that same deterministic rule into the
                # one-call observer: discard the whole element batch instead of
                # clipping coordinates or rejecting an otherwise useful fresh
                # post-action scene.  With no elements this observation cannot
                # authorize another semantic tap; it can only support local
                # transition verification or a coordinate-free action.
                _discard_compact_elements_for_targeted_geometry_recovery(
                    scene_payload,
                    context,
                )
            fused_input_attestation = (
                _fused_preliminary_input_attestation(
                    scene_payload,
                    goal_context=context,
                )
                if input_structure_required
                else None
            )
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
                        fused_input_attestation=fused_input_attestation,
                    ),
                    obstructions,
                    fingerprint=fingerprint,
                )
                if (
                    (
                        _goal_active_input_transaction_text(context)
                        or _goal_has_target_only_active_input_field(context)
                    )
                    and not _input_audit_established_local_target(scene)
                    and not _focus_only_input_surface_established(scene)
                    and not _goal_is_fused_post_action_next_step(context)
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
                "post_action_visual_context": (
                    post_action_context.to_dict()
                    if post_action_context is not None
                    else None
                ),
                "selected_frame_index": selected_frame_index,
                "stable_tail_start_index": stable_tail_start,
                "local_stability": stability.to_dict(),
                "frame_sharpness_scores": [
                    round(value, 3) for value in sharpness_scores
                ],
                "frame_size": list(frame.size),
                "request_image_size": list(request_image_size),
                "coordinate_normalization": envelope.get(
                    "coordinate_normalization"
                ),
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
                "post_action_visual_context": (
                    post_action_context.to_dict()
                    if post_action_context is not None
                    else None
                ),
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
    "chinese_pinyin或unknown。QWERTY只描述按键排列，绝不等于英文键入模式：画面出现中文候选、"
    "拼音分词撇号或明确中文模式时必须写chinese_pinyin；只有明确显示英文/Latin按键模式时才能写"
    "direct_latin；看不清写unknown。direct_latin只描述按键模式，不证明字母已提交到App输入框；"
    "若仍有预编辑和候选，必须由后续输入结构审计分别报告。这些都只是画面事实，不授权输入。"
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


































def _single_step_observation_prompt(
    context: dict[str, Any],
    *,
    include_input_structure: bool,
    current_input_text: str | None,
    image_count: int,
    request_image_size: tuple[int, int],
    post_action_context: PostActionVisualContext | None,
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
    request_width, request_height = request_image_size
    if post_action_context is None:
        post_action_rule = (
            "本轮不是本地已签发的动作后观察；不得猜测此前执行过任何动作。"
        )
    else:
        post_action_payload = json.dumps(
            post_action_context.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        post_action_rule = f"""
本轮是一个物理动作后的首次观察。本地只读typed摘要如下：
{post_action_payload}
它只证明该canonical动作已到达物理执行层，并说明需要核对的typed后置条件；
outcome=pending_visual_verification，绝不等于matched，也不能迫使你把预期写成事实。
当前JPEG仍是当前画面的唯一权威：符合时报告可见结果，不符合时如实报告矛盾。
识别时先判断最外层系统/App表面，再判断其中嵌入的卡片、预览或子内容；嵌入内容
所属App不能替代承载它的最外层系统表面。该摘要只帮助选择核对重点，不授权动作。
"""
    return f"""
这是本闭环步骤唯一一次Qwen视觉调用。你必须在同一个JSON响应中完成当前
画面理解、目标相关事实标记以及必要的输入/IME/键盘结构报告。不得要求第二次
精查、App身份审计、几何审计、方向审计或动作选择调用；不确定时保留unknown、
省略候选或降低confidence。你只报告事实，不输出动作、计划或坐标点击建议。
{temporal_rule}
{post_action_rule}

本轮每张实际发送给你的JPEG均为{request_width}×{request_height}。整份响应的
scene与input_structure必须共用一个coordinate_space，绝不能各用一把尺子：
1. 正常且首选输出为
   {{"kind":"normalized_1000","width":1000,"height":1000}}，此时横纵两轴
   都把各自图像边缘表示为0和1000，任何坐标不得超过1000。
2. 如果你的视觉系统已经在一个与{request_width}×{request_height}严格等比例的
   图像网格中测量了全部坐标，且无法在输出前完成归一化，才可声明
   {{"kind":"image_grid","width":该网格精确宽度,"height":该网格精确高度}}。
   scene和input_structure的每个边界与锚点都必须属于这一个声明网格；本地会在
   解析任何视觉事实前一次性换算为0..1000。禁止猜测1920/2000等常见高度，
   禁止混用normalized、JPEG像素、手机逻辑像素或裁剪坐标。

下面的SCENE CONTRACT和INPUT CONTRACT沿用既有字段语义。它们各自末尾的
“只返回/Return exactly”示例仅说明对应内层对象，不是本轮顶层输出格式。

--- SCENE CONTRACT ---
{scene_contract}

--- INPUT CONTRACT ---
{input_contract}

最终且唯一有效的顶层格式如下，禁止Markdown、重复键和任何额外字段：
{{"protocol_version":"{SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION}",
"coordinate_space":{{"kind":"normalized_1000","width":1000,"height":1000}},
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


def _normalize_single_step_wire_coordinates(
    payload: dict[str, Any],
    *,
    request_image_size: tuple[int, int],
) -> dict[str, Any]:
    """Atomically normalize one declared wire grid into canonical 0..1000.

    The raw image grid is syntax only.  It never reaches UIScene, canonical
    candidates, scope, Controller, or the robot.  A non-canonical grid is
    accepted only when its declared aspect ratio independently agrees with the
    exact JPEG sent in this request and every coordinate belongs to that one
    grid.  No conventional phone resolution is guessed.
    """

    coordinate_space = payload.get("coordinate_space")
    if not isinstance(coordinate_space, dict) or set(coordinate_space) != {
        "kind",
        "width",
        "height",
    }:
        raise UISceneError("单步观察必须声明唯一coordinate_space。")
    kind = coordinate_space.get("kind")
    width = coordinate_space.get("width")
    height = coordinate_space.get("height")
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or not 64 <= width <= 8192
        or not 64 <= height <= 8192
    ):
        raise UISceneError("coordinate_space宽高必须是64..8192整数。")
    if kind not in {"normalized_1000", "image_grid"}:
        raise UISceneError("coordinate_space.kind无效。")
    if kind == "normalized_1000" and (width, height) != (1000, 1000):
        raise UISceneError("normalized_1000必须声明1000×1000。")

    bounds_refs: list[tuple[dict[str, Any], str]] = []
    point_refs: list[tuple[dict[str, Any], str]] = []

    def add_bounds(owner: Any, key: str = "bounds") -> None:
        if isinstance(owner, dict) and owner.get(key) is not None:
            bounds_refs.append((owner, key))

    def add_point(owner: Any, key: str) -> None:
        if isinstance(owner, dict) and owner.get(key) is not None:
            point_refs.append((owner, key))

    scene = payload.get("scene")
    if isinstance(scene, dict):
        for item in scene.get("elements") or []:
            add_bounds(item)
    input_payload = payload.get("input_structure")
    if isinstance(input_payload, dict):
        for item in input_payload.get("application_inputs") or []:
            add_bounds(item)
            if isinstance(item, dict):
                add_bounds(item.get("right_button"))
        for region in input_payload.get("ime_preedit_regions") or []:
            add_bounds(region)
            if isinstance(region, dict):
                for candidate in region.get("candidates") or []:
                    add_bounds(candidate)
        keyboard = input_payload.get("keyboard")
        if isinstance(keyboard, dict):
            add_bounds(keyboard)
            for key in (
                "mode_switch",
                "backspace_key",
                "enter_key",
                "case_switch",
            ):
                add_bounds(keyboard.get(key))
            for collection_name in ("literal_keys", "layout_switches"):
                for item in keyboard.get(collection_name) or []:
                    add_bounds(item)
            anchors = keyboard.get("qwerty_anchors")
            if isinstance(anchors, dict):
                for key in anchors:
                    add_point(anchors, key)

    parsed_bounds: list[tuple[dict[str, Any], str, tuple[float, float, float, float]]] = []
    parsed_points: list[tuple[dict[str, Any], str, tuple[float, float]]] = []
    all_coordinates: list[float] = []
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
            raise UISceneError("coordinate_space内存在无效bounds。")
        left, top, right, bottom = (float(part) for part in value)
        parsed_bounds.append((owner, key, (left, top, right, bottom)))
        all_coordinates.extend((left, top, right, bottom))
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
            raise UISceneError("coordinate_space内存在无效锚点。")
        x, y = (float(part) for part in value)
        parsed_points.append((owner, key, (x, y)))
        all_coordinates.extend((x, y))

    if kind == "normalized_1000":
        # Keep the existing downstream distinction: an input target with
        # invalid canonical geometry fails closed, while a non-input scene may
        # discard its entire unusable element batch and retain only the typed
        # top-level App/screen facts.  This branch never changes coordinates.
        payload.pop("coordinate_space", None)
        return {
            "wire_kind": kind,
            "wire_extent": [1000, 1000],
            "canonical_extent": [1000, 1000],
            "applied": False,
        }

    request_width, request_height = request_image_size
    if request_width <= 0 or request_height <= 0:
        raise UISceneError("本轮Qwen请求图片尺寸无效。")
    if not math.isclose(
        width / height,
        request_width / request_height,
        rel_tol=0.01,
        abs_tol=0.0,
    ):
        raise UISceneError("image_grid宽高比与本轮Qwen请求图片不一致。")
    if not all_coordinates or not any(value > 1000 for value in all_coordinates):
        raise UISceneError("image_grid没有可证明需要归一化的越界坐标。")

    transformed_bounds: list[tuple[dict[str, Any], str, list[int]]] = []
    transformed_points: list[tuple[dict[str, Any], str, list[int]]] = []
    for owner, key, (left, top, right, bottom) in parsed_bounds:
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise UISceneError("bounds超出声明的image_grid。")
        normalized = [
            round(left * 1000 / width),
            round(top * 1000 / height),
            round(right * 1000 / width),
            round(bottom * 1000 / height),
        ]
        if not (
            0 <= normalized[0] < normalized[2] <= 1000
            and 0 <= normalized[1] < normalized[3] <= 1000
        ):
            raise UISceneError("image_grid换算后bounds退化。")
        transformed_bounds.append((owner, key, normalized))
    for owner, key, (x, y) in parsed_points:
        if not (0 <= x <= width and 0 <= y <= height):
            raise UISceneError("锚点超出声明的image_grid。")
        normalized = [round(x * 1000 / width), round(y * 1000 / height)]
        if not (0 <= normalized[0] <= 1000 and 0 <= normalized[1] <= 1000):
            raise UISceneError("image_grid换算后锚点无效。")
        transformed_points.append((owner, key, normalized))

    for owner, key, value in transformed_bounds:
        owner[key] = value
    for owner, key, value in transformed_points:
        owner[key] = value

    scene_inputs = [
        item
        for item in (scene.get("elements") or [])
        if isinstance(item, dict)
        and item.get("role") == "input"
        and isinstance(item.get("states"), dict)
        and item["states"].get("goal_relevant") is True
        and isinstance(item.get("bounds"), list)
    ] if isinstance(scene, dict) else []
    audited_inputs = [
        item
        for item in (input_payload.get("application_inputs") or [])
        if isinstance(item, dict) and isinstance(item.get("bounds"), list)
    ] if isinstance(input_payload, dict) else []
    if scene_inputs and audited_inputs:
        def overlap_ratio(first: list[Any], second: list[Any]) -> float:
            left = max(float(first[0]), float(second[0]))
            top = max(float(first[1]), float(second[1]))
            right = min(float(first[2]), float(second[2]))
            bottom = min(float(first[3]), float(second[3]))
            intersection = max(0.0, right - left) * max(0.0, bottom - top)
            first_area = (float(first[2]) - float(first[0])) * (
                float(first[3]) - float(first[1])
            )
            second_area = (float(second[2]) - float(second[0])) * (
                float(second[3]) - float(second[1])
            )
            return intersection / max(1.0, min(first_area, second_area))

        if not any(
            overlap_ratio(scene_item["bounds"], audit_item["bounds"]) >= 0.5
            for scene_item in scene_inputs
            for audit_item in audited_inputs
        ):
            raise UISceneError(
                "scene与input_structure没有使用同一个image_grid坐标空间。"
            )
    payload.pop("coordinate_space", None)
    return {
        "wire_kind": kind,
        "wire_extent": [width, height],
        "request_image_size": [request_width, request_height],
        "canonical_extent": [1000, 1000],
        "applied": True,
    }


def _parse_single_step_observation_envelope(
    raw: str,
    *,
    input_structure_required: bool,
    request_image_size: tuple[int, int],
) -> dict[str, Any]:
    """Parse one fused response without any remote repair or resampling."""

    try:
        payload = _extract_json_object(raw, reject_duplicate_keys=True)
        # Flat scene JSON remains a valid one-call response for non-input
        # observations.  This is a shape normalization only; it never enables
        # a second request or restores any former audit authority.
        if payload.get("protocol_version") == UI_SCENE_PROTOCOL_VERSION:
            if input_structure_required:
                raise UISceneError("输入子目标的单次响应缺少input_structure。")
            return {
                "scene": payload,
                "input_structure": None,
                "coordinate_normalization": {
                    "wire_kind": "implicit_normalized_1000",
                    "wire_extent": [1000, 1000],
                    "canonical_extent": [1000, 1000],
                    "applied": False,
                },
            }
        required = {
            "protocol_version",
            "coordinate_space",
            "scene",
            "input_structure",
        }
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
        coordinate_normalization = _normalize_single_step_wire_coordinates(
            payload,
            request_image_size=request_image_size,
        )
        input_payload = payload["input_structure"]
        if input_structure_required and not isinstance(input_payload, dict):
            raise UISceneError("输入子目标必须在同一响应返回input_structure对象。")
        if (
            input_structure_required
            and isinstance(input_payload, dict)
            and set(input_payload)
            == {"application_inputs", "ime_preedit_regions", "keyboard"}
        ):
            # The outer single-step envelope already fixes the sole valid
            # nested audit protocol.  Restoring that one omitted constant is a
            # deterministic shape normalization; no visual fact, geometry or
            # action meaning is inferred.  Every other missing/extra/wrong
            # nested field still fails in the strict audit parser below.
            input_payload = {
                "protocol_version": INPUT_STRUCTURE_AUDIT_VERSION,
                **input_payload,
            }
            payload = {**payload, "input_structure": input_payload}
        if not input_structure_required and input_payload is not None:
            raise UISceneError("非输入子目标的input_structure必须为null。")
        return {
            "scene": payload["scene"],
            "input_structure": payload["input_structure"],
            "coordinate_normalization": coordinate_normalization,
        }
    except (UISceneError, ValueError, TypeError) as exc:
        raise VisionAgentError(f"单步完整观察结果不符合协议：{exc}") from exc


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
1. 桌面写 launcher；系统最近任务页面必须写foreground_app_id=system、
   screen_id=system_recent_tasks；不确定写 unknown。不得把目标App当成当前App，也不得把
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
18. 当前画面是launcher、目标App图标尚未出现，且画面有两个或更多分页圆点并能明确看出恰好一个
    当前圆点时，必须把承载桌面图标的完整分页区域额外报告为role=container、
    meaning=paged_viewport。states必须包含goal_relevant:true、fully_visible:true、scrollable:true、
    scroll_axis:"horizontal"、从0开始的page_index和page_count；evidence写明实际看见的圆点总数和
    当前第几页。圆点不清、选中项不唯一或只有一页时不得输出page_index/page_count，也不得猜测。

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
    target_only_clear = _goal_has_target_only_active_input_field(context)
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
1. application_inputs: editable search/address/form fields and message composers in the App content area. Include a visibly empty field when its complete input surface is visible. Literal editable evidence may be its complete border, a distinct fill/perimeter that separates the whole field surface from surrounding App chrome, a placeholder, caret, focus highlight, or one IME preedit visibly rendered inside that complete surface. A complete blank surface does not need placeholder text, a caret, or focus highlight. Never infer a field from the goal, an unexplained gap, or adjacent icons alone. field_labels must contain only literal labels visibly attached to that field (for example a nearby form label or its placeholder), never the local field_id. The active field selector is field_id={json.dumps(active_field_id, ensure_ascii=False)} and visible field_label={json.dumps(active_field_label, ensure_ascii=False)}; target_only_clear={str(target_only_clear).lower()} means the typed graph intentionally supplies no new text payload and asks only to audit the one currently focused field for clearing. It does not relax the visual evidence rules: enumerate that field only when its complete App surface plus focus/preedit evidence are visible, and never invent its bounds from the goal. Use the label only to enumerate visible evidence, never infer it from the goal.
2. ime_preedit_regions: the input method's composition and its candidate strip. It is never an application input, even when it contains composed text and a trailing icon. Many real IMEs render an underlined Latin composition inside the otherwise empty App field. In that layout the underlined letters remain IME preedit, application_inputs.text MUST be "", the literal may also appear in visible_editable_cues, and one ime_preedit_regions item MUST tightly bound the underlined composition with text set to that literal. Candidate words use their own complete bounds and may be either immediately adjacent to the composition or in one horizontal candidate row at the top of the visible keyboard, above the QWERTY letter rows. Never call those underlined letters committed application text. Enumerate only complete visible candidate words tied to that composition; candidates are read-only facts and never application inputs. The enum name direct_latin describes the keyboard key mode only: it NEVER proves that Latin letters bypass composition or are already committed.
3. keyboard.mode_switch: one compact key inside the visible keyboard that explicitly switches between chinese_pinyin and direct_latin. Ordinary letters, backspace, enter, robot/assistant, voice, emoji, and candidate-strip icons are never mode switches.
4. keyboard.qwerty_anchors: only for a complete visible QWERTY keyboard, locate the centers of q, p, a, l, z, m and backspace. These are read-only current-frame geometry facts, not a tap plan. Use null for every non-QWERTY, incomplete or uncertain keyboard.
   When keyboard.visible=true, report keyboard.bounds only when it confidently encloses the complete visible keyboard in the same coordinate system, has width at least 300 and height at least 180, and contains every reported keyboard key and anchor. Measure from the four edges of Image 1; do not shift the keyboard toward the bottom or describe only its letter rows. For QWERTY, qwerty_anchors remain mandatory; when the outer bounds cannot be measured confidently, set bounds=null instead of inventing it. Local code may reconstruct an execution envelope only after independent multi-frame row evidence validates all seven anchors. Non-QWERTY actionable geometry still requires complete keyboard.bounds.
5. keyboard.backspace_key: for any complete visible keyboard layout, report the one complete backspace/delete key as label, bounds, confidence and fully_visible. Use null when absent, clipped, ambiguous, or confused with an App delete control. This is read-only geometry and never authorizes clearing by itself.
6. keyboard.literal_keys: the local, goal-derived whitelist is {json.dumps(literal_key_targets, ensure_ascii=False, separators=(',', ':'))}. Report only complete visible keys whose inserted value occurs in that exact whitelist, at most once per distinct value and at most eight total. Every literal-key object MUST contain exactly these six fields and never omit any of them: value, label, key_kind, bounds, confidence, fully_visible. When the whitelist is empty, literal_keys MUST be []. QWERTY alphabet letters and Chinese characters MUST NEVER be enumerated here, even when they occur in input_text, because qwerty_anchors and the verified pinyin transaction already represent them. Never enumerate a keyboard row. For a whitelisted space use value=" " and key_kind="space". For every other whitelisted key use key_kind="character" and require label to equal value literally. The large central PRIMARY glyph of the whole directly tappable key MUST equal value. A small corner glyph, superscript digit, alternate symbol, swipe hint or long-press hint printed on an alphabet key is NOT a literal key and MUST NEVER be reported here. If the whitelisted value exists only as such a secondary hint, leave literal_keys empty and report a separately visible direction-explicit numeric/symbol layout switch instead. Bounds must enclose the whole direct key, never only the secondary glyph. Never include backspace, enter, send/search, emoji, voice, assistant, shift, or layout switches.
7. keyboard.enter_key: report at most one complete visible keyboard action key using exactly label, bounds, confidence, fully_visible and key_action. key_action must be one of newline, send, search, done, next, unknown and must describe the key's current visible behavior, never the requested goal. A plain multiline Return/Enter key may be newline. A key visibly labelled or iconographically acting as Send/Search/Done/Next must use that action and can never authorize a newline. The current transaction needs a newline={str(enter_required).lower()} and multiline={str(active_multiline).lower()}, but those facts do not change the visual classification.
8. keyboard.layout_switches: enumerate every compact visible key with an explicit destination layout: qwerty, numeric, or symbol. Copy the literal label and report current_layout and target_layout; never infer a destination from the goal alone. In particular, on QWERTY report both a visible 123 key targeting numeric and a separately visible ！？# / !?# / symbol key targeting symbol. Never substitute 123 for a symbol-layout key.
7. keyboard.case_mode and keyboard.case_switch apply only to direct_latin QWERTY. case_mode is lower, upper, or unknown from the visible letter glyphs. case_switch is null unless a complete visible shift/case key and its lower↔upper direction are independently clear.
The local controller has one deterministic keyboard routing policy: Latin letters require QWERTY plus direct_latin, followed by an exact candidate click whenever the resulting letters remain in preedit; Chinese requires QWERTY plus chinese_pinyin and then an exact candidate; decimal digits require the visible 123/numeric layout and then the exact digit; every other printable symbol requires direct_latin first and then the separately visible symbol-layout switch such as ！？# before the exact symbol key. This policy does not authorize an action. It tells you which current state and visible controls must be reported completely so local typed code can select exactly one next action after a fresh observation.
Determine keyboard.input_mode only from the current whole keyboard image, never from the goal, the JSON example, or the mode-switch key label alone. Visible Chinese composition/candidates or pinyin separators prove chinese_pinyin. A plain Latin QWERTY state with no Chinese composition/candidate strip may prove direct_latin only when the whole keyboard provides independent current-mode evidence. If the whole keyboard does not prove the current mode, use unknown and set mode_switch to null.
When a visible preedit composition itself exactly matches a complete visible candidate, that exact candidate MUST be enumerated with its own bounds. This applies equally to direct_latin and chinese_pinyin. In particular, when the same Latin glyph sequence appears once as underlined composition in the App field and again as a separate non-underlined word in the candidate strip above the QWERTY rows, the second occurrence is the exact candidate and MUST have its own candidate bounds. Omitting that second occurrence while reporting the matching preedit is an incomplete audit; never silently turn useful target text into a clear/delete instruction.
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
"layout_switches":[{{"label":"123","bounds":[0,0,1000,1000],"confidence":0.0,"current_layout":"qwerty","target_layout":"numeric"}},{{"label":"！？#","bounds":[0,0,1000,1000],"confidence":0.0,"current_layout":"qwerty","target_layout":"symbol"}}]}}}}
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


def _snap_qwerty_bottom_row_switches_from_local_rows(
    payload: dict[str, Any],
    *,
    locally_snapped_anchors: dict[str, list[int]],
) -> None:
    """Anchor bottom-row switches to the independently observed QWERTY grid.

    A model can place a compact bottom-row switch too close to Android's
    navigation bar. Preserve its audited horizontal bounds and height, but move
    only a switch already reported below the third letter row to the implied
    fourth-row center. This is generic keyboard geometry, not an App patch.
    """

    keyboard = payload.get("keyboard")
    if not isinstance(keyboard, dict) or keyboard.get("layout") != "qwerty":
        return
    try:
        top_y = statistics.mean(
            float(locally_snapped_anchors[key][1]) for key in ("q", "p")
        )
        middle_y = statistics.mean(
            float(locally_snapped_anchors[key][1]) for key in ("a", "l")
        )
        bottom_y = statistics.mean(
            float(locally_snapped_anchors[key][1])
            for key in ("z", "m", "backspace")
        )
    except (KeyError, TypeError, ValueError):
        return
    pitches = (middle_y - top_y, bottom_y - middle_y)
    if (
        min(pitches) < 25
        or max(pitches) > 140
        or max(pitches) / min(pitches) > 1.35
    ):
        return
    pitch = statistics.mean(pitches)
    target_center_y = bottom_y + pitch

    def snap(owner: Any) -> None:
        if not isinstance(owner, dict) or not _valid_1000_bounds(owner.get("bounds")):
            return
        left, top, right, bottom = (float(part) for part in owner["bounds"])
        center_y = (top + bottom) / 2.0
        height = bottom - top
        if not (
            bottom_y + 0.2 * pitch <= center_y <= bottom_y + 2.0 * pitch
            and 0.3 * pitch <= height <= 1.5 * pitch
        ):
            return
        snapped_top = target_center_y - height / 2.0
        snapped_bottom = target_center_y + height / 2.0
        if not 0 <= snapped_top < snapped_bottom <= 1000:
            return
        owner["bounds"] = [
            round(left),
            round(snapped_top),
            round(right),
            round(snapped_bottom),
        ]

    snap(keyboard.get("mode_switch"))
    for item in keyboard.get("layout_switches") or []:
        if isinstance(item, dict) and item.get("current_layout") == "qwerty":
            snap(item)


def _snap_bottom_row_layout_switches_above_system_navigation(
    payload: dict[str, Any],
    *,
    navigation_bar_visible: bool,
) -> None:
    """Keep audited bottom-row layout controls above system navigation.

    The input audit uses a portrait-normalized 0..1000 grid.  When Android's
    navigation area is visible, the last 5.5 percent is not a reliable input
    target.  Preserve an already audited switch's horizontal bounds and height,
    but translate only a bottom-row box that crosses that reserved area.
    """

    if not navigation_bar_visible:
        return
    keyboard = payload.get("keyboard")
    if not isinstance(keyboard, dict):
        return
    safe_bottom = 945.0
    for item in keyboard.get("layout_switches") or []:
        if (
            not isinstance(item, dict)
            or not _valid_1000_bounds(item.get("bounds"))
        ):
            continue
        left, top, right, bottom = (
            float(part) for part in item["bounds"]
        )
        height = bottom - top
        if not (
            top >= 850.0
            and bottom > safe_bottom
            and 25.0 <= height <= 120.0
        ):
            continue
        snapped_top = safe_bottom - height
        if snapped_top < 0:
            continue
        item["bounds"] = [
            round(left),
            round(snapped_top),
            round(right),
            round(safe_bottom),
        ]


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


def _fused_preliminary_input_attestation(
    payload: dict[str, Any],
    *,
    goal_context: dict[str, Any],
) -> dict[str, Any] | None:
    """Keep one non-authoritative empty-field fact from the fused scene.

    The dedicated input structure remains the sole value and text geometry
    owner. This record proves only that the same envelope described one visible,
    goal-bound edit surface with non-empty visual evidence. It can either rescue
    an empty field when the audit reports matching structure, or be sanitized
    into a focus-only surface when the audit misses and the keyboard is hidden.
    """

    if not (
        _goal_active_input_transaction_text(goal_context)
        or _goal_has_target_only_active_input_field(goal_context)
    ):
        return None
    elements = payload.get("elements")
    if not isinstance(elements, list):
        return None
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
    candidates: list[dict[str, Any]] = []
    for item in elements:
        if not isinstance(item, dict) or set(item) != exact_fields:
            continue
        states = item.get("states")
        evidence = item.get("evidence")
        confidence = item.get("confidence")
        raw_bounds = item.get("bounds")
        if (
            str(item.get("role") or "").strip() != "input"
            or not isinstance(item.get("element_id"), str)
            or not item["element_id"].strip()
            or str(item["element_id"]).startswith("local_audited_")
            or not isinstance(item.get("meaning"), str)
            or not item["meaning"].strip()
            or not isinstance(item.get("label"), str)
            or not isinstance(states, dict)
            or states.get("goal_relevant") is not True
            or states.get("fully_visible") is not True
            or not isinstance(states.get("value"), str)
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or float(confidence) < 0.9
            or not isinstance(evidence, list)
            or not evidence
            or any(not isinstance(value, str) or not value.strip() for value in evidence)
            or not _valid_1000_bounds(raw_bounds)
        ):
            continue
        bounds = [float(value) for value in raw_bounds]
        if max(bounds) <= 1.0:
            bounds = [value * 1000.0 for value in bounds]
        candidates.append(
            {
                "element_id": item["element_id"].strip(),
                "meaning": item["meaning"].strip(),
                "label": item["label"].strip()[:200],
                "value": states["value"],
                "bounds": bounds,
                "confidence": float(confidence),
                "evidence": tuple(str(value).strip()[:200] for value in evidence),
            }
        )
    return candidates[0] if len(candidates) == 1 else None


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




















def _goal_requests_input(context: dict[str, Any]) -> bool:
    focused = _active_subgoal_visual_context(context)
    if _goal_has_target_only_active_input_field(context):
        # A clear-only graph deliberately carries no desired text.  Its typed
        # target marker still requires the same one-call application/IME/
        # keyboard audit as ordinary input so the focused field cannot be
        # replaced by a bare candidate word or an unbound backspace key.
        return True
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


def _focus_only_input_surface_established(scene: UIScene) -> bool:
    """Return true only for one locally sanitized coarse focus surface.

    This state is deliberately weaker than a dedicated input audit: it may
    authorize one focus tap, but it carries no value, typed field identity,
    keyboard state, or permission to input, clear, or send text.
    """

    candidates = tuple(
        element
        for element in scene.elements
        if element.role == "input"
        and element.states.get("focus_only_input_surface") is True
        and element.states.get("goal_relevant") is True
        and element.states.get("fully_visible") is True
        and float(element.confidence) >= MIN_TARGET_CONFIDENCE
        and any(str(item).strip() for item in element.evidence)
    )
    if len(candidates) != 1:
        return False
    unique = scene.unique_trusted_goal_element()
    return unique is not None and unique.element_id == candidates[0].element_id


def _focus_only_compact_input_surface(
    attestation: dict[str, Any] | None,
    *,
    keyboard_visible: bool,
) -> dict[str, Any] | None:
    """Sanitize one compact input into non-text focus authority only.

    The compact model transcription is never copied. A visible keyboard means
    focus has progressed far enough that the dedicated audit must now establish
    the typed field; another coarse tap could only move the caret or hide the
    real failure.
    """

    if keyboard_visible or not isinstance(attestation, dict):
        return None
    if set(attestation) != {
        "element_id",
        "meaning",
        "label",
        "value",
        "bounds",
        "confidence",
        "evidence",
    }:
        return None
    element_id = str(attestation.get("element_id") or "").strip()
    meaning = str(attestation.get("meaning") or "").strip()
    label = str(attestation.get("label") or "").strip()[:200]
    bounds = attestation.get("bounds")
    evidence = attestation.get("evidence")
    confidence = attestation.get("confidence")
    if (
        not element_id
        or element_id.startswith("local_audited_")
        or not meaning
        or not isinstance(bounds, (list, tuple))
        or len(bounds) != 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= float(value) <= 1000
            for value in bounds
        )
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or float(confidence) < MIN_TARGET_CONFIDENCE
        or not isinstance(evidence, (list, tuple))
        or not evidence
        or any(not isinstance(item, str) or not item.strip() for item in evidence)
    ):
        return None
    return {
        "element_id": element_id,
        "role": "input",
        "meaning": meaning,
        "bounds": [float(value) / 1000.0 for value in bounds],
        "confidence": float(confidence),
        "label": label,
        "states": {
            "goal_relevant": True,
            "fully_visible": True,
            "focus_only_input_surface": True,
        },
        "evidence": [str(item).strip()[:200] for item in evidence],
    }


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


def _goal_is_fused_post_action_next_step(context: dict[str, Any]) -> bool:
    """Return whether this response also previews one typed successor."""

    focused = _active_subgoal_visual_context(context)
    if focused is context:
        return False
    entities = focused.get("goal_entities")
    return bool(
        isinstance(entities, dict)
        and entities.get("observation_phase")
        == FUSED_POST_ACTION_NEXT_STEP_OBSERVATION_PHASE
    )


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


def _goal_has_target_only_active_input_field(context: dict[str, Any]) -> bool:
    """Return whether the bridge minted one clear-only typed field target."""

    focused = _active_subgoal_visual_context(context)
    if focused is context:
        return False
    entities = focused.get("goal_entities")
    field_id, _field_label, _multiline = _goal_active_input_field(context)
    return bool(
        isinstance(entities, dict)
        and entities.get("active_input_target_only") is True
        and field_id
    )


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
        or keyboard_input_mode not in {"direct_latin", "chinese_pinyin"}
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
    fused_input_attestation: dict[str, Any] | None = None,
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
                _snap_qwerty_bottom_row_switches_from_local_rows(
                    payload,
                    locally_snapped_anchors=locally_snapped_qwerty_anchors,
                )
                _reattach_input_audit_to_unique_scene_field(
                    scene,
                    application_inputs,
                    goal_context=goal_context,
                )
        _snap_bottom_row_layout_switches_above_system_navigation(
            payload,
            navigation_bar_visible=(
                scene.system_ui.navigation_bar_visible is True
            ),
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
            fused_empty_field_evidence: tuple[str, ...] = ()
            if (
                len(application_inputs) == 1
                and text == ""
                and isinstance(fused_input_attestation, dict)
                and fused_input_attestation.get("value") == text
            ):
                attested_bounds = fused_input_attestation.get("bounds")
                attested_evidence = fused_input_attestation.get("evidence")
                if (
                    isinstance(attested_bounds, (list, tuple))
                    and len(attested_bounds) == 4
                    and all(
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        for value in attested_bounds
                    )
                    and isinstance(attested_evidence, (list, tuple))
                    and attested_evidence
                ):
                    audit_bounds = tuple(float(value) for value in item["bounds"])
                    scene_bounds = tuple(float(value) for value in attested_bounds)
                    if (
                        _bounds_overlap_ratio(audit_bounds, scene_bounds) >= 0.85
                        and _bounds_overlap_ratio(scene_bounds, audit_bounds) >= 0.85
                    ):
                        fused_empty_field_evidence = tuple(
                            str(value).strip()[:200]
                            for value in attested_evidence
                            if str(value).strip()
                        )
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
                and not fused_empty_field_evidence
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
                    "fused_empty_field_evidence": fused_empty_field_evidence,
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
            if lineage_matches_trailing_newline_cue(
                verified_input_lineage,
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
            elif lineage_matches_persisted_surface_cue(
                verified_input_lineage,
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
                and lineage_matches_visual(
                    verified_input_lineage,
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
            and (active_transaction_text or active_clear_goal)
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
        required_input_mode = (
            required_keyboard_input_mode_for_step(input_step)
            if input_step is not None
            else None
        )
        input_needs_mode_switch = bool(
            not active_clear_goal
            and not clearable_ime_preedit
            and exact_ime_candidate is None
            and input_step is not None
            and required_input_mode is not None
            and keyboard_visible
            and keyboard_layout == "qwerty"
            and keyboard_input_mode in {"direct_latin", "chinese_pinyin"}
            and keyboard_input_mode != required_input_mode
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
            or mode_switch["target_mode"] != required_input_mode
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
                and not _literal_key_overlaps_qwerty_backspace(
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
                required_input_mode is not None
                and keyboard_input_mode != required_input_mode
            ):
                # Mode switching is available only on the alphabetic surface.
                # From another layout, first return to QWERTY. On QWERTY the
                # audited Chinese/English switch is the only legal next action;
                # do not expose a layout or literal key at the same time.
                if keyboard_layout != "qwerty":
                    exact_layout_switch = (
                        _select_keyboard_layout_switch_for_target(
                            layout_switches,
                            current_layout=keyboard_layout,
                            target_layout="qwerty",
                        )
                    )
            elif (
                input_step.kind in {"direct_latin", "chinese_pinyin"}
                and keyboard_layout != "qwerty"
            ):
                exact_layout_switch = _select_keyboard_layout_switch_for_target(
                    layout_switches,
                    current_layout=keyboard_layout,
                    target_layout="qwerty",
                )
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
                    if desired_layout != keyboard_layout:
                        exact_layout_switch = (
                            _select_keyboard_layout_switch_for_target(
                                layout_switches,
                                current_layout=keyboard_layout,
                                target_layout=desired_layout,
                            )
                        )
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

        focus_only_input = None
        if (
            trusted_input is None
            and next_field_key is None
            and (mode_switch is None or not switch_is_goal)
            and (
                _goal_active_input_transaction_text(goal_context)
                or _goal_has_target_only_active_input_field(goal_context)
            )
        ):
            focus_only_input = _focus_only_compact_input_surface(
                fused_input_attestation,
                keyboard_visible=keyboard_visible,
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
            if focus_only_input is not None:
                elements.append(focus_only_input)
            value["elements"] = elements
            value["summary"] = (
                "专用typed输入状态账本尚未建立；"
                "仅保留唯一粗编辑面用于聚焦，不授权文字、清空或发送。"
                if focus_only_input is not None
                else "typed输入状态账本未建立；compact输入摘要与输入转写不参与判断。"
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
                        required_input_mode is not None
                        and keyboard_input_mode != required_input_mode
                    )
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
                        *rendered_input.get("fused_empty_field_evidence", ()),
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
            elif rendered_input.get("fused_empty_field_evidence"):
                input_evidence.insert(
                    0,
                    "同一单步响应的场景输入事实与输入结构审计唯一重合；"
                    "当前输入框为空",
                )
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
    try:
        return preferred_keyboard_layout(character)
    except VerifiedTextTransactionError as exc:
        raise UISceneError(str(exc)) from exc


def _select_keyboard_layout_switch_for_target(
    layout_switches: list[dict[str, Any]],
    *,
    current_layout: str,
    target_layout: str,
) -> dict[str, Any] | None:
    """Select one explicit switch that monotonically approaches a layout.

    The local layout graph is a three-state line: numeric <-> qwerty <->
    symbol. A visible direct edge wins. When no direct edge is visible, only
    the unique adjacent edge on the shortest path may be used.  Every edge was
    already read from the current image and validated for its literal label,
    source layout, geometry and confidence; this helper never invents a key.
    """

    if (
        next_keyboard_layout_towards(current_layout, target_layout) is None
    ):
        return None
    direct = [
        item
        for item in layout_switches
        if item.get("current_layout") == current_layout
        and item.get("target_layout") == target_layout
    ]
    if len(direct) == 1:
        return direct[0]
    if direct:
        return None
    next_layout = next_keyboard_layout_towards(
        current_layout,
        target_layout,
    )
    next_hop = [
        item
        for item in layout_switches
        if item.get("current_layout") == current_layout
        and item.get("target_layout") == next_layout
    ]
    return next_hop[0] if len(next_hop) == 1 else None


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
            # A schema-valid optional key with a mismatched visible glyph is
            # not executable (for example ASCII "?" claimed as full-width
            # "？"). Revoke only this candidate so the independently audited
            # input value and a valid layout switch can still be consumed.
            continue
        if key_kind == "space" and (
            key_value != " "
            or label.strip().casefold() not in {"", "space", "空格"}
        ):
            continue
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


def _literal_key_overlaps_qwerty_backspace(
    item: dict[str, Any],
    *,
    qwerty_geometry: dict[str, Any],
) -> bool:
    """Reject a claimed literal key whose box owns the audited backspace.

    QWERTY anchors independently identify the current frame's backspace key.
    A model-authored punctuation box containing that anchor is therefore a
    conflicting identity claim, even when its printed label matches the next
    requested character.  The candidate is removed so the existing generic
    layout-switch path can expose the character on its real keyboard layer.
    """

    bounds = item.get("bounds")
    anchors = qwerty_geometry.get("anchors")
    if (
        not isinstance(bounds, list)
        or len(bounds) != 4
        or not isinstance(anchors, dict)
    ):
        return True
    backspace = anchors.get("backspace")
    if not isinstance(backspace, (list, tuple)) or len(backspace) != 2:
        return True
    try:
        left, top, right, bottom = (float(part) for part in bounds)
        backspace_x, backspace_y = (float(part) for part in backspace)
    except (TypeError, ValueError):
        return True
    return bool(
        left <= backspace_x <= right
        and top <= backspace_y <= bottom
    )


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

    # This object is optional, read-only structural context and never grants
    # action authority.  Providers commonly include the ordinary visibility
    # fact on an otherwise schema-valid adjacent control.  Once geometry proves
    # the control is separate from the input, discard the whole object instead
    # of allowing that harmless optional fact to veto the input observation.
    allowed = {"label", "bounds", "confidence", "fully_visible"}
    if (
        not isinstance(input_item, dict)
        or not isinstance(value, dict)
        or not set(value).issubset(allowed)
        or "bounds" not in value
        or (
            "fully_visible" in value
            and not isinstance(value.get("fully_visible"), bool)
        )
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
        and _valid_1000_bounds(item.get("bounds"))
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
    # confirmed by the canonical candidate and local controller binding before
    # an element-bound action.


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
        item["states"].pop("independent_geometry_verified", None)
        item["states"].pop("geometry_audit_source", None)
        item["states"].pop("focus_only_input_surface", None)
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







def _observation_cache_key(
    *,
    device_id: str | None,
    fingerprint: str,
    goal_context: dict[str, Any],
    input_lineage: TypedInputLineage | None,
    post_action_context: PostActionVisualContext | None,
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
        "post_action_context": (
            post_action_context.to_dict()
            if post_action_context is not None
            else None
        ),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
