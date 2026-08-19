from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
import re
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageChops, ImageStat

from generic_intent import GenericIntentDraft
from generic_scene_observer import (
    GenericSceneObserver,
    POST_NAVIGATION_RESULT_COMPLETION_CONDITIONS,
    POST_NAVIGATION_RESULT_OBJECTIVE,
    POST_NAVIGATION_RESULT_OBSERVATION_PHASE,
)
from input_value_lineage import (
    InputValueLineageError,
    TypedInputLineage,
    TypedInputLineageStore,
    build_pending_literal_lineage,
)
from ocr_runtime import recognize as recognize_ocr
from observation_images import measure_local_stability
from orientation_safety import (
    OrientationCredential,
    OrientationFrameMismatchError,
    OrientationSafetyError,
    frame_fingerprint,
    validate_device_id,
)
from qwen_runtime_errors import FORMAT_ERROR_TYPES, classify_qwen_error
from semantic_executor import SemanticAction
from ui_scene import UIElement, UIScene, UISceneError
from universal_action_controller import (
    ResolvedSemanticAction,
    UniversalActionController,
    UniversalActionError,
    navigation_semantic_class,
)
from robot_core import WorkflowNotReady, qwerty_keyboard_config_from_anchors


QWEN_FAILURE_DIAGNOSTIC_VERSION = "2026-08-17-qwen-failure-diagnostic-v1"
MAX_REDACTED_QWEN_RESPONSE_CHARS = 16000
_IMAGE_DATA_URL_RE = re.compile(
    r"data:image/[^;\s\"']+;base64,[A-Za-z0-9+/=_-]+",
    re.IGNORECASE,
)
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

_POST_NAVIGATION_RESULT_KINDS = frozenset(
    {"tap_semantic", "swipe", "back", "home"}
)
_POST_NAVIGATION_ALLOWED_EFFECT_KEYS = frozenset(
    {
        "scene_changed",
        "content_changed",
        "current_video_changed",
        "goal_complete_on_success",
        "description",
        "app_id",
        "screen_id",
    }
)


def _post_action_observation_context(
    goal: GenericIntentDraft,
    resolved: ResolvedSemanticAction,
) -> dict[str, Any]:
    """Return a result-focused context only for a proven navigation boundary."""

    context = goal.to_dict()
    entities = context.get("entities")
    focus = (
        entities.get("active_subgoal_visual_context")
        if isinstance(entities, dict)
        else None
    )
    if isinstance(entities, dict) and isinstance(focus, dict):
        supplied_goal_entities = focus.get("goal_entities")
        if isinstance(supplied_goal_entities, dict):
            # ``observation_phase`` is a reserved local attestation.  Strip any
            # model/user supplied value before deciding whether this resolved
            # action is allowed to mint it.
            sanitized_goal_entities = dict(supplied_goal_entities)
            sanitized_goal_entities.pop("observation_phase", None)
            sanitized_focus = dict(focus)
            sanitized_focus["goal_entities"] = sanitized_goal_entities
            sanitized_entities = dict(entities)
            sanitized_entities["active_subgoal_visual_context"] = sanitized_focus
            context = dict(context)
            context["entities"] = sanitized_entities
            entities = sanitized_entities
            focus = sanitized_focus
    expected = resolved.expected_effect
    if (
        not isinstance(focus, dict)
        or str(focus.get("external_impact") or "").strip() != "navigation_only"
        or resolved.kind not in _POST_NAVIGATION_RESULT_KINDS
        or not isinstance(expected, dict)
        or expected.get("scene_changed") is not True
        or expected.get("goal_complete_on_success") is not True
        or "element_state" in expected
        or set(expected) - _POST_NAVIGATION_ALLOWED_EFFECT_KEYS
    ):
        return context

    goal_entities = focus.get("goal_entities")
    if not isinstance(goal_entities, dict):
        return context
    result_entities = dict(goal_entities)
    result_entities.pop("target_ui_label", None)
    result_entities["observation_phase"] = (
        POST_NAVIGATION_RESULT_OBSERVATION_PHASE
    )
    result_focus = dict(focus)
    result_focus["objective"] = POST_NAVIGATION_RESULT_OBJECTIVE
    result_focus["completion_conditions"] = list(
        POST_NAVIGATION_RESULT_COMPLETION_CONDITIONS
    )
    result_focus["goal_entities"] = result_entities
    result_context = dict(context)
    result_context["entities"] = dict(entities)
    result_context["entities"]["active_subgoal_visual_context"] = result_focus
    return result_context


def stable_qwerty_ocr_anchors(
    frames: tuple[Image.Image, ...] | list[Image.Image],
    anchors: dict[str, Any],
    *,
    ocr_recognizer: Any = recognize_ocr,
) -> dict[str, list[int]] | None:
    """Snap QWERTY row heights to stable local OCR glyph centers.

    Qwen supplies the semantic keyboard contract and coarse row endpoints.
    Local OCR contributes only the three vertical row centers. It cannot add
    characters, choose text, or authorize an input action.
    """

    frame_list = list(frames)[-3:]
    if len(frame_list) != 3 or not isinstance(anchors, dict):
        return None
    try:
        qwerty_keyboard_config_from_anchors(anchors)
        original = {
            key: [round(float(value[0])), round(float(value[1]))]
            for key, value in anchors.items()
            if isinstance(value, (list, tuple)) and len(value) == 2
        }
        if set(original) != {"q", "p", "a", "l", "z", "m", "backspace"}:
            return None
        per_frame_rows: list[tuple[float, float]] = []
        top_letters = set("qwertyuiop")
        bottom_letters = set("zxcvbnm")
        for frame in frame_list:
            payload = ocr_recognizer(
                frame.convert("RGB"),
                "zh-Hans-CN",
                scale=3.0,
            )
            expected_top = frame.height * (
                (original["q"][1] + original["p"][1]) / 2000.0
            )
            expected_bottom = frame.height * (
                (original["z"][1] + original["m"][1]) / 2000.0
            )
            tolerance = frame.height * 0.08
            top_hits: dict[str, float] = {}
            bottom_hits: dict[str, float] = {}
            for line in payload.get("lines") or []:
                for word in line.get("words") or []:
                    text = str(word.get("text") or "").strip().casefold()
                    if len(text) != 1 or not text.isascii() or not text.isalpha():
                        continue
                    center_y = float(word.get("top", 0)) + float(
                        word.get("height", 0)
                    ) / 2.0
                    if text in top_letters and abs(center_y - expected_top) <= tolerance:
                        top_hits[text] = center_y
                    if (
                        text in bottom_letters
                        and abs(center_y - expected_bottom) <= tolerance
                    ):
                        bottom_hits[text] = center_y
            if len(top_hits) < 2 or len(bottom_hits) < 2:
                return None
            top_y = float(statistics.median(top_hits.values()))
            bottom_y = float(statistics.median(bottom_hits.values()))
            gap = bottom_y - top_y
            if not frame.height * 0.08 <= gap <= frame.height * 0.22:
                return None
            per_frame_rows.append((top_y, bottom_y))

        if (
            max(item[0] for item in per_frame_rows)
            - min(item[0] for item in per_frame_rows)
            > 10
            or max(item[1] for item in per_frame_rows)
            - min(item[1] for item in per_frame_rows)
            > 10
        ):
            return None
        height = frame_list[-1].height
        top_y = round(1000 * statistics.median(item[0] for item in per_frame_rows) / height)
        bottom_y = round(
            1000 * statistics.median(item[1] for item in per_frame_rows) / height
        )
        middle_y = round((top_y + bottom_y) / 2.0)
        if (
            abs(top_y - original["q"][1]) > 90
            or abs(middle_y - original["a"][1]) > 90
            or abs(bottom_y - original["z"][1]) > 90
        ):
            return None
        snapped = {key: list(value) for key, value in original.items()}
        for key in ("q", "p"):
            snapped[key][1] = top_y
        for key in ("a", "l"):
            snapped[key][1] = middle_y
        for key in ("z", "m", "backspace"):
            snapped[key][1] = bottom_y
        qwerty_keyboard_config_from_anchors(snapped)
        return snapped
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
    redacted = _UNQUOTED_SECRET_FIELD_RE.sub(
        lambda match: f"{match.group('prefix')}[REDACTED_SECRET]",
        redacted,
    )
    redacted = _BEARER_RE.sub("Bearer [REDACTED_SECRET]", redacted)
    redacted = _URL_SECRET_RE.sub(
        lambda match: f"{match.group('prefix')}[REDACTED_SECRET]",
        redacted,
    )
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
            diagnostics["diagnostic_persistence_error"] = type(
                diagnostic_error
            ).__name__
        return ()
    return (str(path),) if path is not None else ()


class GenericActionAdapterError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        physical_actions: int = 0,
        evidence: tuple[str, ...] = (),
        observation_errors: tuple[str, ...] = (),
        verification_errors: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.physical_actions = int(physical_actions)
        self.evidence = tuple(evidence)
        self.observation_errors = tuple(observation_errors)
        self.verification_errors = tuple(verification_errors)


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
    action_outcome: str = "matched"
    verification_errors: tuple[str, ...] = ()
    robot_result: Any = None
    hardware_receipt: dict[str, Any] | None = None
    evidence: tuple[str, ...] = ()
    after_frames: tuple[Image.Image, ...] = field(
        default_factory=tuple,
        repr=False,
        compare=False,
    )
    after_frame_paths: tuple[str, ...] = ()
    observation_errors: tuple[str, ...] = ()
    controller_completion_evidence: tuple[str, ...] = ()
    before_frames: tuple[Image.Image, ...] = field(
        default_factory=tuple,
        repr=False,
        compare=False,
    )
    before_frame_paths: tuple[str, ...] = ()
    orientation_credential: OrientationCredential | None = None

    def __post_init__(self) -> None:
        # Capture helpers intentionally build mutable lists while sampling.  The
        # completed execution result is a live promotion source, so freeze both
        # frame collections at the result boundary instead of exposing a
        # list-shaped object that the promotion validator must reject.
        object.__setattr__(self, "after_frames", tuple(self.after_frames))
        object.__setattr__(self, "before_frames", tuple(self.before_frames))
        object.__setattr__(
            self,
            "controller_completion_evidence",
            tuple(self.controller_completion_evidence),
        )

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
            "controller_completion_evidence": list(
                self.controller_completion_evidence
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
            "swipe",
            "reveal_system_navigation",
            "back",
            "home",
            "input_verified_text",
            "clear_verified_text",
            "long_press",
            "drag",
        }
    )
    GEOMETRY_BOUND_KINDS = frozenset(
        {
            "tap_semantic",
            "dismiss_overlay",
            "input_verified_text",
            "clear_verified_text",
            "long_press",
            "drag",
        }
    )
    INDEPENDENT_GEOMETRY_AUDIT_KINDS = frozenset(
        {
            "tap_semantic",
            "dismiss_overlay",
            "input_verified_text",
            "clear_verified_text",
            "long_press",
            "drag",
        }
    )
    LOCAL_INPUT_AUXILIARY_MEANINGS = frozenset(
        {
            "ime_exact_candidate",
            "input_exact_literal_key",
            "switch_keyboard_layout",
            "switch_keyboard_case",
            "switch_keyboard_input_mode",
        }
    )

    @staticmethod
    def _has_local_independent_geometry_attestation(
        scene: UIScene,
        element_ids: tuple[str, ...],
    ) -> bool:
        try:
            elements = tuple(scene.get_element(element_id) for element_id in element_ids)
        except UISceneError:
            return False
        return bool(
            elements
            and all(
                element.states.get("independent_geometry_verified") is True
                and element.states.get("geometry_audit_source")
                in {
                    "icon_cluster_localization",
                    "element_geometry_audit",
                }
                and any(item.strip() for item in element.evidence)
                for element in elements
            )
        )

    def supported_action_kinds(self) -> frozenset[str]:
        """Return only actions backed by callable methods on this device."""

        capability_provider = getattr(self.robot, "hardware_capabilities", None)
        declared = capability_provider() if callable(capability_provider) else {}
        if not isinstance(declared, dict):
            declared = {}

        def available(action: str, method_name: str) -> bool:
            return callable(getattr(self.robot, method_name, None)) and bool(
                declared.get(action, True)
            )

        supported = (
            {"wait_for_change"}
            if bool(declared.get("wait_for_change", True))
            else set()
        )
        if available("tap_semantic", "vision_tap_relative"):
            supported.add("tap_semantic")
        if available("dismiss_overlay", "vision_dismiss_overlay_relative"):
            supported.add("dismiss_overlay")
        if bool(declared.get("swipe", True)) and any(
            callable(getattr(self.robot, f"vision_swipe_{direction}", None))
            for direction in ("up", "down", "left", "right")
        ):
            supported.add("swipe")
        if available(
            "reveal_system_navigation",
            "vision_reveal_system_navigation",
        ):
            supported.add("reveal_system_navigation")
        if available("back", "vision_android_back"):
            supported.add("back")
        if available("home", "vision_android_home"):
            supported.add("home")
        if available("input_verified_text", "vision_type_text_with_layout"):
            supported.add("input_verified_text")
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
        from action_capabilities import build_device_capability_snapshot

        provider = getattr(self.robot, "hardware_capability_profile", None)
        raw_profile = provider() if callable(provider) else None
        return build_device_capability_snapshot(
            device_id=str(getattr(self.robot, "device_id", "") or "unknown-device"),
            supported_actions=self.supported_action_kinds(),
            raw_profile=raw_profile,
        )

    @classmethod
    def _local_input_auxiliary_recovery_target(
        cls,
        requested: SemanticAction,
        planned_scene: UIScene,
        fresh_scene: UIScene,
    ) -> UIElement | None:
        """Return a strict local input descriptor omitted by a fresh full pass.

        Ordinary buttons never use this path.  The target must have been minted
        by the canonical input-structure audit, remain fully bound to one
        focused input and be completely absent from the fresh semantic scene.
        Its geometry is still re-audited on the fresh confirmation frames
        before the controller may resolve an action.
        """

        if requested.action != "tap_semantic" or not str(
            requested.params.get("formal_candidate_id") or ""
        ).strip():
            return None
        target_id = str(requested.params.get("element_id") or "").strip()
        try:
            target = planned_scene.get_element(target_id)
        except UISceneError:
            return None
        states = target.states
        marker_by_meaning = {
            "ime_exact_candidate": "ime_candidate",
            "input_exact_literal_key": "input_literal_key",
            "switch_keyboard_layout": "keyboard_layout_switch",
            "switch_keyboard_case": "keyboard_case_switch",
            "switch_keyboard_input_mode": "keyboard_input_mode_switch",
        }
        marker = marker_by_meaning.get(target.meaning)
        if (
            target.meaning not in cls.LOCAL_INPUT_AUXILIARY_MEANINGS
            or target.role not in {"button", "icon"}
            or not target.label.strip()
            or marker is None
            or states.get(marker) is not True
            or states.get("goal_relevant") is not True
            or states.get("fully_visible") is not True
            or requested.params.get("target") != target.meaning
            or requested.params.get("role") != target.role
            or requested.params.get("label") != target.label
            or requested.params.get("states") != states
        ):
            return None
        input_id = str(states.get("input_element_id") or "").strip()
        prior_value = states.get("prior_input_value")
        try:
            input_element = planned_scene.get_element(input_id)
        except UISceneError:
            return None
        if (
            input_element.role != "input"
            or input_element.states.get("focused") is not True
            or not isinstance(prior_value, str)
            or input_element.states.get("value") != prior_value
        ):
            return None
        expected = requested.params.get("expected_effect")
        expected_element = (
            expected.get("element_state")
            if isinstance(expected, dict)
            else None
        )
        expected_states = (
            expected_element.get("states")
            if isinstance(expected_element, dict)
            else None
        )
        if (
            not isinstance(expected_element, dict)
            or expected_element.get("meaning") != input_element.meaning
            or not isinstance(expected_states, dict)
            or not isinstance(expected_states.get("value"), str)
        ):
            return None
        # A visible same-label control is not an omission.  It must go through
        # the ordinary semantic rebinding path so meaning/role/state drift or
        # duplication remains fail-closed.
        if any(
            element.label == target.label
            and element.role in {"button", "icon"}
            for element in fresh_scene.elements
        ):
            return None
        return target

    def capability_gap(
        self,
        requested_action: str,
        *,
        required_parameters: tuple[str, ...] = (),
    ) -> Any:
        return self.capability_snapshot().gap(
            requested_action,
            required_parameters=required_parameters,
        )

    def __init__(
        self,
        *,
        capture: Callable[[], Image.Image],
        observer: GenericSceneObserver,
        robot: Any,
        controller: UniversalActionController | None = None,
        frame_interval: float = 0.5,
        post_action_settle: float = 1.5,
        post_action_timeout: float = 10.0,
        post_action_max_observations: int = 2,
        confirmation_frame_delta_max: float = 6.0,
        qwerty_row_snapper: Callable[
            [tuple[Image.Image, ...] | list[Image.Image], dict[str, Any]],
            dict[str, Any] | None,
        ]
        | None = None,
        require_local_qwerty_row_snap: bool = False,
        device_id: str,
        input_lineage_store: TypedInputLineageStore | None = None,
    ) -> None:
        self.capture = capture
        self.observer = observer
        self.robot = robot
        self.controller = controller or UniversalActionController()
        self.frame_interval = max(0.0, float(frame_interval))
        self.post_action_settle = max(0.0, float(post_action_settle))
        self.post_action_timeout = max(0.0, float(post_action_timeout))
        self.post_action_max_observations = min(
            2,
            max(1, int(post_action_max_observations)),
        )
        self.confirmation_frame_delta_max = max(
            0.0,
            float(confirmation_frame_delta_max),
        )
        self.qwerty_row_snapper = qwerty_row_snapper
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
            raise GenericActionAdapterError(
                f"确认前本地多帧稳定性检查未通过：{stability.reason}",
                evidence=paths,
            )
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
            raise GenericActionAdapterError(
                f"通用页面观察失败：{exc}",
                evidence=paths + diagnostic_paths,
            ) from exc
        return scene, frames, paths

    def _observe_scene(
        self,
        frames: list[Image.Image] | tuple[Image.Image, ...],
        goal_context: dict[str, Any],
        *,
        input_lineage_override: TypedInputLineage | None = None,
    ) -> UIScene:
        kwargs: dict[str, Any] = {
            "frames": list(frames),
            "goal_context": goal_context,
        }
        if getattr(self.observer, "input_lineage_store", None) is not None:
            kwargs["device_id"] = self.device_id
            kwargs["input_lineage_override"] = input_lineage_override
        return self.observer.observe(**kwargs)

    def capture_scene(
        self,
        goal: GenericIntentDraft,
        *,
        evidence_dir: Path | None,
        prefix: str,
    ) -> tuple[UIScene, list[Image.Image], tuple[str, ...]]:
        all_paths: tuple[str, ...] = ()
        errors: list[str] = []
        for attempt in range(1, 3):
            try:
                scene, frames, paths = self._capture_scene_once(
                    goal,
                    evidence_dir=evidence_dir,
                    prefix=f"{prefix}_attempt_{attempt}",
                )
                return scene, frames, all_paths + paths
            except GenericActionAdapterError as exc:
                all_paths += tuple(exc.evidence)
                errors.append(f"第{attempt}轮动作前观察失败：{exc}")
                if attempt >= 2 or not self._pre_action_observation_retryable(exc):
                    raise GenericActionAdapterError(
                        "动作前通用页面观察失败：" + "；".join(errors),
                        evidence=all_paths,
                        observation_errors=tuple(errors),
                    ) from exc
        raise AssertionError("unreachable")

    def _pre_action_observation_retryable(self, error: Exception) -> bool:
        text = str(error)
        return self._post_observation_retryable(error) or any(
            marker in text
            for marker in (
                "页面不稳定",
                "画面不稳定",
                "整体置信度不足",
                "不能建立可信候选",
            )
        )

    def _capture_stable_post_action_frames(
        self,
        *,
        deadline: float,
        evidence_dir: Path | None,
        prefix: str,
    ) -> tuple[list[Image.Image], tuple[str, ...]]:
        """Wait for four consecutive locally stable frames within the deadline.

        This gate is deliberately local and cheap.  Qwen is called only after
        the camera's outer/static UI bands have settled, and no physical action
        is ever repeated while waiting.
        """

        frames: list[Image.Image] = []
        last_stability = None
        while True:
            frames.append(self._capture_frame())
            if len(frames) > 4:
                frames.pop(0)
            if len(frames) == 4:
                last_stability = measure_local_stability(frames)
                if last_stability.stable:
                    paths = self._save_frames(frames, evidence_dir, prefix)
                    return list(frames), paths
                if time.monotonic() >= deadline:
                    paths = self._save_frames(
                        frames,
                        evidence_dir,
                        f"{prefix}_timeout",
                    )
                    raise GenericActionAdapterError(
                        "动作后画面在限定时间内没有稳定："
                        f"{last_stability.reason}",
                        evidence=paths,
                    )
            if time.monotonic() >= deadline:
                paths = self._save_frames(
                    frames,
                    evidence_dir,
                    f"{prefix}_timeout",
                )
                reason = (
                    last_stability.reason
                    if last_stability is not None
                    else "未能采集到连续4帧"
                )
                raise GenericActionAdapterError(
                    f"动作后画面在限定时间内没有稳定：{reason}",
                    evidence=paths,
                )
            if self.frame_interval:
                time.sleep(
                    min(
                        self.frame_interval,
                        max(0.0, deadline - time.monotonic()),
                    )
                )

    def _observe_stable_post_action_scene(
        self,
        goal: GenericIntentDraft,
        *,
        before: UIScene,
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
        if self.post_action_settle:
            time.sleep(min(self.post_action_settle, self.post_action_timeout))

        all_paths: tuple[str, ...] = ()
        observation_errors: list[str] = []
        verification_errors: list[str] = []
        last_error: Exception | None = None
        observation_context = _post_action_observation_context(goal, resolved)
        for attempt in range(1, self.post_action_max_observations + 1):
            attempt_deadline = time.monotonic() + self.post_action_timeout
            try:
                frames, paths = self._capture_stable_post_action_frames(
                    deadline=attempt_deadline,
                    evidence_dir=evidence_dir,
                    prefix=f"{evidence_prefix}_after_attempt_{attempt}",
                )
            except GenericActionAdapterError as exc:
                capture_evidence = all_paths + tuple(exc.evidence)
                prior_errors = (
                    "；此前" + "；".join(observation_errors + verification_errors)
                    if observation_errors or verification_errors
                    else ""
                )
                raise GenericActionAdapterError(
                    f"动作后画面采集失败：{exc}{prior_errors}",
                    evidence=capture_evidence,
                    observation_errors=tuple(observation_errors),
                    verification_errors=tuple(verification_errors),
                ) from exc
            all_paths += paths
            try:
                after = self._observe_scene(
                    frames,
                    observation_context,
                    input_lineage_override=input_lineage_override,
                )
            except RuntimeError as exc:
                last_error = exc
                diagnostic_paths = persist_observer_failure_diagnostic(
                    self.observer,
                    evidence_dir=evidence_dir,
                    prefix=f"{evidence_prefix}_after_attempt_{attempt}",
                    error=exc,
                )
                all_paths += diagnostic_paths
                observation_errors.append(
                    f"第{attempt}轮动作后观察失败：{exc}"
                )
                if (
                    attempt >= self.post_action_max_observations
                    or not self._post_observation_retryable(exc)
                ):
                    raise GenericActionAdapterError(
                        "通用页面观察失败："
                        + "；".join(verification_errors + observation_errors),
                        evidence=all_paths,
                        observation_errors=tuple(observation_errors),
                        verification_errors=tuple(verification_errors),
                    ) from exc
                continue

            after = self._reconcile_literal_key_visual_wrap(
                resolved,
                before,
                after,
            )

            try:
                self.controller.verify_after_action(resolved, before, after)
                return (
                    after,
                    tuple(frames),
                    paths,
                    all_paths,
                    tuple(observation_errors),
                    (),
                )
            except UniversalActionError as exc:
                last_error = exc
                verification_errors.append(
                    f"第{attempt}轮动作结果不匹配：{exc}"
                )
                # A stable old page, a low-confidence transitional scene, or
                # an expected destination that is still loading can all be a
                # legitimate intermediate state.  Re-observe at most once,
                # with a fresh bounded capture, without repeating the action.
                if attempt >= self.post_action_max_observations:
                    break
                if self.frame_interval:
                    time.sleep(self.frame_interval)

        assert last_error is not None
        return (
            after,
            tuple(frames),
            paths,
            all_paths,
            tuple(observation_errors),
            tuple(verification_errors),
        )

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
        expected_states = (
            expected_state.get("states")
            if isinstance(expected_state, dict)
            else None
        )
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
            label=(
                expected
                if candidate.label == observed
                else candidate.label
            ),
            states={**candidate.states, "value": expected},
            evidence=candidate.evidence
            + ("本地逐键回执确认该换行为控件视觉软折行",),
        )
        reconciled = replace(
            after,
            elements=tuple(
                replacement if element.element_id == input_id else element
                for element in after.elements
            ),
        )
        reconciled.validate()
        return reconciled

    def _post_observation_retryable(self, error: Exception) -> bool:
        diagnostics = getattr(self.observer, "last_diagnostics", {})
        error_type = (
            diagnostics.get("error_type")
            if isinstance(diagnostics, dict)
            else None
        )
        return (
            error_type in FORMAT_ERROR_TYPES
            or classify_qwen_error(error) in FORMAT_ERROR_TYPES
        )

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
        local_input_consensus_applied = False
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
            if requested_action.action in self.GEOMETRY_BOUND_KINDS:
                try:
                    before = self._observe_scene(before_frames, goal.to_dict())
                except RuntimeError as exc:
                    diagnostic_paths = persist_observer_failure_diagnostic(
                        self.observer,
                        evidence_dir=evidence_dir,
                        prefix=f"{evidence_prefix}_confirmation",
                        error=exc,
                    )
                    raise GenericActionAdapterError(
                        f"确认前目标几何复核失败：{exc}",
                        evidence=before_paths + diagnostic_paths,
                    ) from exc
        else:
            before, before_frames, before_paths = self.capture_scene(
                goal,
                evidence_dir=evidence_dir,
                prefix=f"{evidence_prefix}_before",
            )
        try:
            rebind_planned_scene = planned_scene
            if (
                planned_frames
                and requested_action.action in self.INDEPENDENT_GEOMETRY_AUDIT_KINDS
            ):
                audit_geometry = getattr(
                    self.observer,
                    "audit_element_geometry",
                    None,
                )
                if not callable(audit_geometry):
                    raise GenericActionAdapterError(
                        "当前观察器没有独立目标几何审计，拒绝几何绑定动作。"
                    )
                local_input_recovery = (
                    self._local_input_auxiliary_recovery_target(
                        requested_action,
                        planned_scene,
                        before,
                    )
                )
                if local_input_recovery is not None:
                    planned_ids = (local_input_recovery.element_id,)
                    # The full-scene confirmation pass omitted a model-generated
                    # ordinary keyboard control.  Keep only the old descriptor,
                    # bind it to a fingerprint from the actual fresh stable
                    # frames, and require two independent crop audits before
                    # semantic rebinding.  No old geometry survives this path.
                    fresh_seed_scene = replace(
                        planned_scene,
                        fingerprint=frame_fingerprint(before_frames[-1]),
                    )
                    rebind_planned_scene = audit_geometry(
                        frames=tuple(planned_frames),
                        scene=planned_scene,
                        element_ids=planned_ids,
                    )
                    before = audit_geometry(
                        frames=before_frames,
                        scene=fresh_seed_scene,
                        element_ids=planned_ids,
                    )
                else:
                    semantic_rebound = self._rebind_action(
                        requested_action,
                        planned_scene,
                        before,
                        local_frame_identity_verified=(
                            local_frame_identity_verified
                        ),
                        require_geometry_overlap=False,
                    )
                    if requested_action.action == "drag":
                        planned_ids = (
                            str(
                                requested_action.params.get(
                                    "source_element_id"
                                )
                                or ""
                            ),
                            str(
                                requested_action.params.get(
                                    "destination_element_id"
                                )
                                or ""
                            ),
                        )
                        fresh_ids = (
                            str(
                                semantic_rebound.params.get(
                                    "source_element_id"
                                )
                                or ""
                            ),
                            str(
                                semantic_rebound.params.get(
                                    "destination_element_id"
                                )
                                or ""
                            ),
                        )
                    else:
                        planned_ids = (
                            str(
                                requested_action.params.get("element_id")
                                or ""
                            ),
                        )
                        fresh_ids = (
                            str(
                                semantic_rebound.params.get("element_id")
                                or ""
                            ),
                        )
                    if not (
                        self._has_local_independent_geometry_attestation(
                            planned_scene,
                            planned_ids,
                        )
                        and self._has_local_independent_geometry_attestation(
                            before,
                            fresh_ids,
                        )
                    ):
                        rebind_planned_scene = audit_geometry(
                            frames=tuple(planned_frames),
                            scene=planned_scene,
                            element_ids=planned_ids,
                        )
                        before = audit_geometry(
                            frames=before_frames,
                            scene=before,
                            element_ids=fresh_ids,
                        )
            consensus_scene = self._apply_local_input_geometry_consensus(
                requested_action,
                rebind_planned_scene,
                before,
                local_frame_identity_verified=local_frame_identity_verified,
            )
            local_input_consensus_applied = consensus_scene is not before
            before = consensus_scene
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
            raise GenericActionAdapterError(
                f"确认前独立目标几何审计失败：{exc}",
                evidence=before_paths,
            ) from exc
        try:
            resolved = self.controller.resolve_one(
                rebound,
                before,
                confirmed=True,
            )
        except UniversalActionError as exc:
            raise GenericActionAdapterError(
                f"确认前控制器拒绝动作：{exc}",
                evidence=before_paths,
            ) from exc

        if local_input_consensus_applied and resolved.kind == "tap_semantic":
            resolver = getattr(
                self.robot,
                "resolve_calibrated_target_grid_point",
                None,
            )
            if callable(resolver):
                try:
                    target = before.get_element(
                        str(resolved.target_element_id or ""),
                        min_confidence=self.controller.min_confidence,
                    )
                    if resolved.normalized_point is None:
                        raise GenericActionAdapterError("局部输入目标缺少共识落点。")
                    preferred_x = round(resolved.normalized_point[0] * 1000)
                    preferred_y = round(resolved.normalized_point[1] * 1000)
                    resolved_x, resolved_y = resolver(
                        preferred_x,
                        preferred_y,
                        target.bounds,
                        before_frames[-1].size,
                    )
                    if (
                        isinstance(resolved_x, bool)
                        or isinstance(resolved_y, bool)
                        or not isinstance(resolved_x, int)
                        or not isinstance(resolved_y, int)
                        or not 0 <= resolved_x <= 1000
                        or not 0 <= resolved_y <= 1000
                    ):
                        raise GenericActionAdapterError(
                            "机械标定没有返回合法的局部目标落点。"
                        )
                    resolved = replace(
                        resolved,
                        normalized_point=(resolved_x / 1000.0, resolved_y / 1000.0),
                    )
                except (UISceneError, RuntimeError, ValueError) as exc:
                    raise GenericActionAdapterError(
                        f"局部输入目标与实测标定区域无法形成安全落点：{exc}",
                        evidence=before_paths,
                    ) from exc

        if resolved.kind not in self.PHYSICAL_KINDS and resolved.kind != "wait_for_change":
            raise GenericActionAdapterError(
                f"当前通用硬件适配器尚未开放：{resolved.kind}",
                evidence=before_paths,
            )

        prepared_input_method: Callable[..., Any] | None = None
        prepared_keyboard_geometry: dict[str, Any] | None = None
        if resolved.kind in {"input_verified_text", "clear_verified_text"}:
            if resolved.kind == "input_verified_text" and not resolved.text:
                raise GenericActionAdapterError("输入动作缺少已校验文字。")
            method_name = (
                (
                    "vision_type_pinyin"
                    if resolved.input_method == "chinese_pinyin"
                    else "vision_type_text_with_layout"
                )
                if resolved.kind == "input_verified_text"
                else "vision_clear_text"
            )
            method = getattr(self.robot, method_name, None)
            if not callable(method):
                raise GenericActionAdapterError(
                    "机械臂不支持绑定本轮键盘几何的文字输入或清空。"
                )
            try:
                input_element = before.get_element(
                    str(resolved.target_element_id or ""),
                    min_confidence=self.controller.min_confidence,
                )
            except UISceneError as exc:
                raise GenericActionAdapterError(
                    f"当前文字输入缺少可信输入框：{exc}"
                ) from exc
            keyboard_geometry = input_element.states.get("keyboard_geometry")
            if (
                not isinstance(keyboard_geometry, dict)
                or keyboard_geometry.get("source") != "input_structure_audit"
                or (
                    resolved.kind == "input_verified_text"
                    and keyboard_geometry.get("type") != "qwerty"
                )
                or (
                    resolved.kind == "clear_verified_text"
                    and keyboard_geometry.get("type") not in {"qwerty", "generic"}
                )
            ):
                raise GenericActionAdapterError(
                    "当前文字动作缺少本轮输入结构审计签发的键盘几何；拒绝使用静态配置。"
                )
            execution_keyboard_geometry = dict(keyboard_geometry)
            if (
                keyboard_geometry.get("type") == "qwerty"
                and self.require_local_qwerty_row_snap
            ):
                if not callable(self.qwerty_row_snapper):
                    raise GenericActionAdapterError(
                        "真机文字输入缺少本地 QWERTY 行中心复核器。"
                    )
                snapped_anchors = self.qwerty_row_snapper(
                    before_frames,
                    keyboard_geometry.get("anchors"),
                )
                if not isinstance(snapped_anchors, dict):
                    raise GenericActionAdapterError(
                        "本地 OCR 未能稳定确认 QWERTY 三行中心，拒绝按模型粗坐标输入。"
                    )
                execution_keyboard_geometry["anchors"] = snapped_anchors
                execution_keyboard_geometry["row_snap_source"] = "stable_local_ocr"
            if execution_keyboard_geometry.get("type") == "qwerty":
                try:
                    qwerty_keyboard_config_from_anchors(
                        execution_keyboard_geometry.get("anchors")
                    )
                except WorkflowNotReady as exc:
                    raise GenericActionAdapterError(
                        f"当前 QWERTY 几何未通过动作前本地复核：{exc}"
                    ) from exc
            else:
                backspace = (
                    execution_keyboard_geometry.get("anchors") or {}
                ).get("backspace")
                if (
                    not isinstance(backspace, list)
                    or len(backspace) != 2
                    or any(
                        isinstance(part, bool)
                        or not isinstance(part, (int, float))
                        or not 0 <= float(part) <= 1000
                        for part in backspace
                    )
                ):
                    raise GenericActionAdapterError(
                        "非 QWERTY 清空缺少本轮完整可见退格键中心。"
                    )
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
                    raise GenericActionAdapterError(
                        f"当前文字输入不满足设备已验证配置：{exc}"
                    ) from exc
            if resolved.kind == "clear_verified_text" and resolved.delete_count is None:
                raise GenericActionAdapterError("清空动作缺少已验证退格次数。")
            prepared_input_method = method
            prepared_keyboard_geometry = execution_keyboard_geometry

        orientation_credential: OrientationCredential | None = None
        clear_authorization = getattr(
            self.robot, "clear_physical_execution_authorization", None
        )
        if resolved.kind in self.PHYSICAL_KINDS:
            arm = getattr(self.robot, "arm_physical_execution", None)
            audit = (
                getattr(
                    self.observer,
                    "audit_coordinate_free_system_navigation_alignment",
                    None,
                )
                if resolved.kind == "home"
                else getattr(self.observer, "audit_camera_alignment", None)
            )
            if resolved.kind == "home" and not callable(audit):
                audit = getattr(self.observer, "audit_camera_alignment", None)
            if not callable(arm) or not callable(clear_authorization):
                raise GenericActionAdapterError(
                    "机械臂控制器未提供共享物理执行门禁，拒绝动作。",
                    evidence=before_paths,
                )
            if not callable(audit):
                raise GenericActionAdapterError(
                    "观察器未提供独立方向审计，拒绝动作。",
                    evidence=before_paths,
                )
            clear_authorization()
            try:
                orientation_credential = audit(
                    frames=before_frames,
                    device_id=self.device_id,
                    scene_fingerprint=before.fingerprint,
                )
                audit_diagnostics = getattr(
                    self.observer,
                    "last_orientation_audit_diagnostics",
                    {},
                )
                selected_index = (
                    audit_diagnostics.get("selected_frame_index")
                    if isinstance(audit_diagnostics, dict)
                    else None
                )
                if (
                    isinstance(selected_index, int)
                    and 0 <= selected_index < len(before_paths)
                ):
                    with Image.open(before_paths[selected_index]) as persisted:
                        orientation_credential = replace(
                            orientation_credential,
                            evidence_frame_fingerprint=frame_fingerprint(
                                persisted.convert("RGB")
                            ),
                        )
                orientation_credential.assert_authorizes(
                    device_id=self.device_id,
                    scene_fingerprint=before.fingerprint,
                    frame_size=orientation_credential.frame_size,
                )
                hardware_action = (
                    "input_verified_text"
                    if resolved.kind == "clear_verified_text"
                    else resolved.kind
                )
                arm(
                    orientation_credential,
                    action=hardware_action,
                    scene_fingerprint=before.fingerprint,
                )
            except (OrientationSafetyError, RuntimeError, ValueError) as exc:
                clear_authorization()
                raise GenericActionAdapterError(
                    f"动作前独立方向凭据校验失败：{exc}",
                    evidence=before_paths,
                ) from exc

        physical_actions = 0
        robot_result: Any = None
        hardware_receipt: dict[str, Any] | None = None
        try:
            if resolved.kind in {"tap_semantic", "dismiss_overlay"}:
                if resolved.normalized_point is None:
                    raise GenericActionAdapterError("点击动作缺少已校验落点。")
                x = max(0, min(1000, round(resolved.normalized_point[0] * 1000)))
                y = max(0, min(1000, round(resolved.normalized_point[1] * 1000)))
                physical_actions = 1
                if resolved.kind == "dismiss_overlay":
                    robot_result = self.robot.vision_dismiss_overlay_relative(x, y)
                else:
                    robot_result = self.robot.vision_tap_relative(x, y)
            elif resolved.kind == "reveal_system_navigation":
                method = getattr(
                    self.robot,
                    "vision_reveal_system_navigation",
                    None,
                )
                if not callable(method):
                    raise GenericActionAdapterError(
                        "机械臂不支持经过验证的系统导航栏唤出动作。"
                    )
                physical_actions = 1
                robot_result = method()
            elif resolved.kind == "swipe":
                method = getattr(self.robot, f"vision_swipe_{resolved.direction}", None)
                if not callable(method):
                    raise GenericActionAdapterError(
                        f"机械臂不支持滑动方向：{resolved.direction}"
                    )
                physical_actions = 1
                robot_result = method()
            elif resolved.kind == "back":
                physical_actions = 1
                robot_result = self.robot.vision_android_back()
            elif resolved.kind == "home":
                physical_actions = 1
                robot_result = self.robot.vision_android_home()
            elif resolved.kind in {"input_verified_text", "clear_verified_text"}:
                if (
                    prepared_input_method is None
                    or prepared_keyboard_geometry is None
                ):
                    raise GenericActionAdapterError("文字动作的本地预检结果缺失。")
                physical_actions = 1
                if resolved.kind == "input_verified_text":
                    if resolved.input_method == "chinese_pinyin":
                        robot_result = prepared_input_method(
                            resolved.input_fragment,
                            resolved.input_pinyin,
                            prepared_keyboard_geometry,
                        )
                    else:
                        robot_result = prepared_input_method(
                            resolved.input_fragment,
                            prepared_keyboard_geometry,
                        )
                else:
                    robot_result = prepared_input_method(
                        prepared_keyboard_geometry,
                        resolved.delete_count,
                    )
            elif resolved.kind == "long_press":
                if resolved.normalized_point is None or resolved.hold_seconds is None:
                    raise GenericActionAdapterError("长按动作缺少已校验落点或时长。")
                method = getattr(self.robot, "vision_long_press_relative", None)
                if not callable(method):
                    raise GenericActionAdapterError("机械臂不支持通用长按。")
                x = max(0, min(1000, round(resolved.normalized_point[0] * 1000)))
                y = max(0, min(1000, round(resolved.normalized_point[1] * 1000)))
                physical_actions = 1
                robot_result = method(x, y, resolved.hold_seconds)
                receipt_consumer = getattr(
                    self.robot,
                    "consume_last_long_press_receipt",
                    None,
                )
                if not callable(receipt_consumer):
                    raise GenericActionAdapterError(
                        "机械臂没有提供长按事件栅栏凭据。",
                        physical_actions=physical_actions,
                        evidence=before_paths,
                    )
                raw_receipt = receipt_consumer()
                if not isinstance(raw_receipt, dict):
                    raise GenericActionAdapterError(
                        "机械臂没有返回长按事件栅栏凭据。",
                        physical_actions=physical_actions,
                        evidence=before_paths,
                    )
                hardware_receipt = dict(raw_receipt)
            elif resolved.kind == "drag":
                if (
                    resolved.normalized_point is None
                    or resolved.normalized_end_point is None
                ):
                    raise GenericActionAdapterError("拖动动作缺少已校验起点或终点。")
                method = getattr(self.robot, "vision_drag_relative", None)
                if not callable(method):
                    raise GenericActionAdapterError(
                        "当前机械臂控制端没有经过验收的任意拖动能力。"
                    )
                start_x = max(
                    0, min(1000, round(resolved.normalized_point[0] * 1000))
                )
                start_y = max(
                    0, min(1000, round(resolved.normalized_point[1] * 1000))
                )
                end_x = max(
                    0, min(1000, round(resolved.normalized_end_point[0] * 1000))
                )
                end_y = max(
                    0, min(1000, round(resolved.normalized_end_point[1] * 1000))
                )
                physical_actions = 1
                robot_result = method(start_x, start_y, end_x, end_y)
            elif resolved.kind == "wait_for_change":
                time.sleep(max(0.5, self.post_action_settle))
            if resolved.kind in {
                "tap_semantic",
                "dismiss_overlay",
                "back",
                "home",
            }:
                receipt_consumer = getattr(
                    self.robot,
                    "consume_last_click_receipt",
                    None,
                )
                if callable(receipt_consumer):
                    raw_receipt = receipt_consumer()
                    if (
                        not isinstance(raw_receipt, dict)
                        or raw_receipt.get("seller_event_barrier_confirmed") is not True
                        or raw_receipt.get("round_trip_position_confirmed") is not True
                        or raw_receipt.get("mechanical_contact_ack") is not False
                    ):
                        raise GenericActionAdapterError(
                            "机械臂没有返回有效的单击事件栅栏凭据。",
                            physical_actions=physical_actions,
                            evidence=before_paths,
                        )
                    hardware_receipt = dict(raw_receipt)
        except GenericActionAdapterError:
            raise
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

        pending_input_lineage: TypedInputLineage | None = None
        if hardware_receipt is not None:
            try:
                pending_input_lineage = build_pending_literal_lineage(
                    device_id=self.device_id,
                    resolved_action=resolved.to_dict(),
                    before_scene=before.to_dict(),
                    hardware_receipt=hardware_receipt,
                )
            except (InputValueLineageError, TypeError, ValueError):
                pending_input_lineage = None

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
                observation_errors=tuple(
                    getattr(exc, "observation_errors", ())
                ),
                verification_errors=tuple(
                    getattr(exc, "verification_errors", ())
                ),
            ) from exc

        controller_completion_evidence: tuple[str, ...] = ()
        if not verification_errors:
            try:
                controller_completion_evidence = (
                    self.controller.completion_evidence_after_action(
                        resolved,
                        before,
                        after,
                    )
                )
            except UniversalActionError as exc:
                verification_errors = verification_errors + (
                    f"控制器完成证据复核失败：{exc}",
                )

        if (
            not verification_errors
            and self.input_lineage_store is not None
            and hardware_receipt is not None
        ):
            try:
                self.input_lineage_store.record_verified_literal_action(
                    device_id=self.device_id,
                    resolved_action=resolved.to_dict(),
                    before_scene=before.to_dict(),
                    after_scene=after.to_dict(),
                    hardware_receipt=hardware_receipt,
                    after_frames=after_frames,
                )
            except (InputValueLineageError, OSError, TypeError, ValueError):
                # The lineage is only a future read-only disambiguation hint.
                # Failure to persist it must not rewrite a correctly verified
                # physical action, and it never grants action authority.
                pass

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
            action_outcome=(
                "mismatched" if verification_errors else "matched"
            ),
            verification_errors=verification_errors,
            robot_result=robot_result,
            hardware_receipt=hardware_receipt,
            evidence=before_paths + all_after_paths,
            after_frames=after_frames,
            after_frame_paths=after_frame_paths,
            observation_errors=observation_errors,
            controller_completion_evidence=controller_completion_evidence,
            before_frames=before_frames,
            before_frame_paths=before_paths,
            orientation_credential=orientation_credential,
        )

    def _local_input_geometry_consensus_bounds(
        self,
        requested: SemanticAction,
        original: UIElement,
        current: UIElement,
        *,
        prefix: str = "",
        local_frame_identity_verified: bool = False,
    ) -> tuple[float, float, float, float] | None:
        """Return only the rectangle independently attributed by both audits."""

        def stable_states(element: UIElement) -> dict[str, Any]:
            return {
                key: value
                for key, value in element.states.items()
                if key not in {"goal_relevant", "keyboard_geometry"}
            }

        if not (
            local_frame_identity_verified
            and requested.action == "tap_semantic"
            and prefix == ""
            and original.element_id == current.element_id
            and original.element_id.startswith("local_audited_")
            and original.meaning == current.meaning
            and original.meaning in self.LOCAL_INPUT_AUXILIARY_MEANINGS
            and original.role == current.role
            and original.role in {"button", "icon"}
            and original.label == current.label
            and bool(original.label.strip())
            and stable_states(original) == stable_states(current)
            and original.states.get("independent_geometry_verified") is True
            and current.states.get("independent_geometry_verified") is True
            and original.states.get("geometry_audit_source")
            == "element_geometry_audit"
            and current.states.get("geometry_audit_source")
            == "element_geometry_audit"
        ):
            return None
        left = max(original.bounds[0], current.bounds[0])
        top = max(original.bounds[1], current.bounds[1])
        right = min(original.bounds[2], current.bounds[2])
        bottom = min(original.bounds[3], current.bounds[3])
        if not (left < right and top < bottom):
            return None
        intersection = (right - left) * (bottom - top)
        original_width = original.bounds[2] - original.bounds[0]
        original_height = original.bounds[3] - original.bounds[1]
        current_width = current.bounds[2] - current.bounds[0]
        current_height = current.bounds[3] - current.bounds[1]
        smaller_area = min(
            original_width * original_height,
            current_width * current_height,
        )
        smaller_coverage = intersection / smaller_area if smaller_area > 0 else 0.0
        center_delta_x = abs(original.center[0] - current.center[0])
        center_delta_y = abs(original.center[1] - current.center[1])
        narrow_consensus = bool(
            smaller_coverage >= 0.25
            and center_delta_x
            <= max(0.05, 0.30 * max(original_width, current_width))
            and center_delta_y
            <= max(0.03, 0.75 * max(original_height, current_height))
        )
        wider_consensus = bool(
            smaller_coverage >= 0.40
            and center_delta_x
            <= max(0.10, 0.50 * max(original_width, current_width))
            and center_delta_y
            <= max(0.03, 0.75 * max(original_height, current_height))
        )
        return (left, top, right, bottom) if narrow_consensus or wider_consensus else None

    def _apply_local_input_geometry_consensus(
        self,
        requested: SemanticAction,
        planned_scene: UIScene,
        fresh_scene: UIScene,
        *,
        local_frame_identity_verified: bool = False,
    ) -> UIScene:
        if requested.action != "tap_semantic":
            return fresh_scene
        element_id = str(requested.params.get("element_id") or "").strip()
        try:
            original = planned_scene.get_element(element_id)
            current = fresh_scene.get_element(element_id)
        except UISceneError:
            return fresh_scene
        consensus = self._local_input_geometry_consensus_bounds(
            requested,
            original,
            current,
            local_frame_identity_verified=local_frame_identity_verified,
        )
        if consensus is None:
            return fresh_scene
        return replace(
            fresh_scene,
            elements=tuple(
                replace(element, bounds=consensus)
                if element.element_id == element_id
                else element
                for element in fresh_scene.elements
            ),
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
            and
            planned_app != "unknown"
            and fresh_app != "unknown"
            and planned_app != fresh_app
        ):
            raise GenericActionAdapterError(
                f"确认时前台 App 已变化：{planned_app} -> {fresh_app}"
            )
        if (
            not local_frame_identity_verified
            and
            planned_scene.screen_id != "unknown"
            and planned_scene.screen_id != fresh_scene.screen_id
        ):
            raise GenericActionAdapterError(
                f"确认时页面已变化：{planned_scene.screen_id} -> {fresh_scene.screen_id}"
            )
        single_element_actions = {
            "tap_semantic",
            "dismiss_overlay",
            "input_verified_text",
            "clear_verified_text",
            "long_press",
        }
        if requested.action not in single_element_actions | {"drag"}:
            return requested

        def stable_rebind_states(states: dict[str, Any]) -> dict[str, Any]:
            # goal_relevant is a task-context annotation produced by the visual
            # observer, not part of the control's stable identity. A fresh
            # confirmation observation may legitimately omit or recompute it.
            # Keep all physical/actionability attestations fail-closed.
            return {
                key: value
                for key, value in states.items()
                if key not in {"goal_relevant", "keyboard_geometry"}
                and not (
                    key == "keyboard_case_mode"
                    and str(value or "").strip().casefold() == "unknown"
                )
            }

        def compatible_rebind_states(
            original_states: dict[str, Any],
            current_states: dict[str, Any],
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            """Return comparable state views without treating unknown as fact.

            ``keyboard_case_mode=unknown`` is an explicit lack of an
            observation, not a claim that the keyboard has a third case mode.
            A subsequent local audit may resolve it to ``lower`` or ``upper``
            while every action-relevant input precondition remains unchanged.
            Only that originally unknown field is removed from the fresh view;
            known case changes and every other stable state still fail closed.
            """

            original_stable = stable_rebind_states(original_states)
            current_stable = stable_rebind_states(current_states)
            if str(
                original_states.get("keyboard_case_mode") or ""
            ).strip().casefold() == "unknown":
                current_stable.pop("keyboard_case_mode", None)
            return original_stable, current_stable

        def rebind_element(prefix: str = "") -> UIElement:
            original_id = str(
                requested.params.get(f"{prefix}element_id") or ""
            ).strip()
            try:
                original = planned_scene.get_element(original_id)
            except UISceneError as exc:
                raise GenericActionAdapterError(f"原始场景目标无效：{exc}") from exc
            matches = fresh_scene.find_elements(
                label=original.label or None,
                role=original.role,
                states=stable_rebind_states(dict(original.states)),
            )
            # Text-styled links are routinely described as either ``button``
            # or ``text`` across two otherwise identical visual reads.  The
            # exact visible label, stable states, safe navigation class and
            # geometry checks below remain authoritative, so this only keeps
            # a uniquely rebound link from failing on model role wording.
            selector_roles = {"button", "tab", "list_item", "text"}
            if (
                not matches
                and requested.action in {"tap_semantic", "dismiss_overlay"}
                and prefix == ""
                and original.label
                and original.role in selector_roles
            ):
                role_agnostic_matches = fresh_scene.find_elements(
                    label=original.label,
                    states=stable_rebind_states(dict(original.states)),
                )
                if (
                    len(role_agnostic_matches) == 1
                    and role_agnostic_matches[0].role in selector_roles
                ):
                    matches = role_agnostic_matches
            if len(matches) != 1:
                raise GenericActionAdapterError(
                    "确认时目标语义不再严格唯一："
                    f"{original.meaning}，匹配{len(matches)}个"
                )
            current = matches[0]
            if current.meaning.casefold() != original.meaning.casefold():
                if str(
                    requested.params.get("formal_candidate_id") or ""
                ).strip():
                    raise GenericActionAdapterError(
                        "确认时正式候选 meaning 已变化，旧 authority 失效："
                        f"{original.meaning} -> {current.meaning}。"
                    )
                def semantic_class(meaning: str, label: str) -> str:
                    normalized = str(meaning or "").casefold()
                    if requested.action == "drag" and prefix in {
                        "source_",
                        "destination_",
                    }:
                        for marker in (
                            "draggable",
                            "drag_source",
                            "drag_target",
                            "drop_target",
                            "drag",
                            "drop",
                            "拖动",
                            "拖拽",
                        ):
                            normalized = normalized.replace(marker, " ")
                    return navigation_semantic_class(normalized, label)

                original_class = semantic_class(
                    original.meaning,
                    original.label,
                )
                current_class = semantic_class(
                    current.meaning,
                    current.label,
                )
                labelled_drag_endpoint = bool(
                    requested.action == "drag"
                    and prefix in {"source_", "destination_"}
                    and original.label
                    and current.label == original.label
                    and original_class != "forbidden"
                    and current_class != "forbidden"
                )
                def stripped_gesture_semantics(value: str) -> str:
                    normalized = str(value or "").casefold()
                    for marker in (
                        "long_press",
                        "longpress",
                        "drag",
                        "drop",
                        "长按",
                        "拖动",
                        "拖拽",
                    ):
                        normalized = normalized.replace(marker, " ")
                    return normalized

                selector_tokens = {
                    token
                    for token in re.split(
                        r"[^a-z0-9]+",
                        " ".join(
                            (
                                stripped_gesture_semantics(original.meaning),
                                stripped_gesture_semantics(current.meaning),
                            )
                        ),
                    )
                    if token
                }
                selector_semantics = " ".join(
                    (
                        stripped_gesture_semantics(original.meaning),
                        stripped_gesture_semantics(current.meaning),
                    )
                )
                has_selector_semantics = bool(
                    selector_tokens.intersection(
                        {
                            "select",
                            "selector",
                            "mode",
                            "option",
                            "entry",
                            "navigate",
                            "open",
                        }
                    )
                    or any(
                        marker in selector_semantics
                        for marker in (
                            "选择",
                            "选项",
                            "模式",
                            "入口",
                            "进入",
                            "打开",
                            "导航",
                        )
                    )
                )
                labelled_local_mode_selector = bool(
                    requested.action in {"tap_semantic", "dismiss_overlay"}
                    and prefix == ""
                    and original.label
                    and current.label == original.label
                    and original.role in selector_roles
                    and current.role in selector_roles
                    and stable_rebind_states(dict(current.states))
                    == stable_rebind_states(dict(original.states))
                    and has_selector_semantics
                    and navigation_semantic_class(
                        stripped_gesture_semantics(original.meaning),
                        stripped_gesture_semantics(original.label),
                    )
                    != "forbidden"
                    and navigation_semantic_class(
                        stripped_gesture_semantics(current.meaning),
                        stripped_gesture_semantics(current.label),
                    )
                    != "forbidden"
                )
                labelled_long_press_target = bool(
                    requested.action == "long_press"
                    and prefix == ""
                    and original.label
                    and current.label == original.label
                    and original.role == current.role
                    and current.role != "container"
                    and stable_rebind_states(dict(current.states))
                    == stable_rebind_states(dict(original.states))
                    and navigation_semantic_class(
                        stripped_gesture_semantics(original.meaning),
                        stripped_gesture_semantics(original.label),
                    )
                    != "forbidden"
                    and navigation_semantic_class(
                        stripped_gesture_semantics(current.meaning),
                        stripped_gesture_semantics(current.label),
                    )
                    != "forbidden"
                )
                stable_input_field = bool(
                    requested.action in {
                        "tap_semantic", "input_verified_text", "clear_verified_text"
                    }
                    and prefix == ""
                    and original.role == "input"
                    and current.role == "input"
                    and current.label == original.label
                    and compatible_rebind_states(
                        dict(original.states), dict(current.states)
                    )[1]
                    == compatible_rebind_states(
                        dict(original.states), dict(current.states)
                    )[0]
                )
                if not (
                    labelled_drag_endpoint
                    or labelled_local_mode_selector
                    or labelled_long_press_target
                    or stable_input_field
                ) and (
                    not original_class
                    or original_class == "forbidden"
                    or current_class != original_class
                ):
                    raise GenericActionAdapterError(
                        "确认时目标语义已经变化，旧确认失效："
                        f"{original.meaning} -> {current.meaning}。"
                    )
            (
                original_stable_states,
                current_stable_states,
            ) = compatible_rebind_states(
                dict(original.states), dict(current.states)
            )
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
                raise GenericActionAdapterError(
                    "确认时目标标签或状态已经变化，旧确认失效。"
                )
            left = max(original.bounds[0], current.bounds[0])
            top = max(original.bounds[1], current.bounds[1])
            right = min(original.bounds[2], current.bounds[2])
            bottom = min(original.bounds[3], current.bounds[3])
            intersection = max(0.0, right - left) * max(0.0, bottom - top)
            original_area = max(
                0.0, original.bounds[2] - original.bounds[0]
            ) * max(0.0, original.bounds[3] - original.bounds[1])
            current_area = max(
                0.0, current.bounds[2] - current.bounds[0]
            ) * max(0.0, current.bounds[3] - current.bounds[1])
            union = original_area + current_area - intersection
            overlap = intersection / union if union > 0 else 0.0
            smaller_area = min(original_area, current_area)
            smaller_coverage = (
                intersection / smaller_area if smaller_area > 0 else 0.0
            )
            original_width = max(0.0, original.bounds[2] - original.bounds[0])
            original_height = max(0.0, original.bounds[3] - original.bounds[1])
            current_width = max(0.0, current.bounds[2] - current.bounds[0])
            current_height = max(0.0, current.bounds[3] - current.bounds[1])
            center_delta_x = abs(
                (original.bounds[0] + original.bounds[2]) / 2.0
                - (current.bounds[0] + current.bounds[2]) / 2.0
            )
            center_delta_y = abs(
                (original.bounds[1] + original.bounds[3]) / 2.0
                - (current.bounds[1] + current.bounds[3]) / 2.0
            )
            tight_loose_same_target = bool(
                intersection > 0
                and smaller_coverage >= 0.50
                and center_delta_x
                <= max(0.03, 0.25 * max(original_width, current_width))
                and center_delta_y
                <= max(0.02, 0.50 * max(original_height, current_height))
            )
            local_input_consensus_bounds = (
                self._local_input_geometry_consensus_bounds(
                    requested,
                    original,
                    current,
                    prefix=prefix,
                    local_frame_identity_verified=local_frame_identity_verified,
                )
            )
            if (
                require_geometry_overlap
                and overlap < 0.60
                and not tight_loose_same_target
                and local_input_consensus_bounds is None
            ):
                raise GenericActionAdapterError(
                    "确认时目标区域已明显移动，旧确认失效："
                    f"iou={overlap:.3f}, smaller_coverage={smaller_coverage:.3f}, "
                    f"center_delta=({center_delta_x:.3f},{center_delta_y:.3f})。"
                )
            if local_input_consensus_bounds is not None:
                # Neither model crop owns the execution point.  The overlap is
                # the only region independently attributed to the same exact
                # local input target by both audits, so execute at its center.
                return replace(
                    current,
                    bounds=local_input_consensus_bounds,
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
        return SemanticAction(
            node_id=requested.node_id,
            action=requested.action,
            params=params,
        )

    @staticmethod
    def _save_frames(
        frames: list[Image.Image],
        evidence_dir: Path | None,
        prefix: str,
    ) -> tuple[str, ...]:
        if evidence_dir is None:
            return ()
        evidence_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for index, frame in enumerate(frames, start=1):
            path = evidence_dir / f"{prefix}_{index}.jpg"
            frame.save(path, format="JPEG", quality=92)
            paths.append(str(path))
        return tuple(paths)
