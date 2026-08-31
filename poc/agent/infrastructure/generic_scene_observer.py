"""Local visual-observation adapter for the generic Agent runtime."""

from __future__ import annotations

from agent.domain.validation import NormalizedBounds, canonical_digest, reject_if
import json
from functools import lru_cache
from importlib.resources import files
import math
import re
import statistics
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable

from PIL import Image
import agent.domain.post_action_observation as post_action_contract
from agent.infrastructure.observation_images import (
    consensus_top_edge_obstructions,
    local_frame_fingerprint,
    measure_frame_sharpness,
    measure_local_stability,
)
from agent.domain.visual_evidence import VisualObstruction
from agent.domain.canonical_action_kinds import CANONICAL_ACTION_KINDS
from agent.infrastructure.qwen_runtime_errors import classify_qwen_error
from agent.infrastructure.robot_controller import WorkflowNotReady, qwerty_keyboard_config_from_anchors
from agent.domain.ui_scene import (
    CAMERA_LAYOUT_ORIENTATIONS,
    CameraAlignmentFacts,
    MIN_TARGET_CONFIDENCE,
    PHONE_CONTENT_ROTATIONS,
    UI_SCENE_PROTOCOL_VERSION,
    UIScene,
    UISceneError,
    camera_alignment_evidence_is_safe,
)
from agent.infrastructure.dashscope_vision_provider import _extract_json_object, _image_data_url, _image_request_size
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
from agent.domain.input_value_lineage import TypedInputLineage
import agent.domain.generic_goal as generic_goal_domain

SINGLE_STEP_SCENE_OBSERVER_VERSION = "2026-09-01-single-step-scene-action-finish-v5"
SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION = "2026-09-01-single-step-qwen-action-finish-v4"
INPUT_STRUCTURE_AUDIT_VERSION = "2026-08-25-input-structure-audit-v11"
SINGLE_STEP_OUTPUT_TOKENS = 5200
@lru_cache(maxsize=4)
def _prompt_template(name: str) -> str:
    with files(__package__).joinpath("prompts", name).open("r", encoding="utf-8") as stream:
        return stream.read()


def _render_prompt(name: str, **values: str) -> str:
    template = _prompt_template(name)
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    return template
OBSERVATION_TIMEOUT_SECONDS = 60.0
MAX_COMPACT_ELEMENTS = 12
AUDITED_SOFT_KEYBOARD_HIDDEN_EVIDENCE = "输入结构只读审计确认软键盘不可见"

_PASSIVE_SCENE_ELEMENT_FIELDS = frozenset({'element_id', 'role', 'meaning', 'label', 'bounds', 'confidence', 'states',
    'evidence'})
_ACTION_LIKE_WIRE_KEYS = frozenset({'action', 'actions', 'plan', 'step', 'steps', 'tap', 'swipe', 'command',
    'coordinates'})
_MODEL_DECISION_FIELDS = frozenset({'status', 'action', 'element_id', 'source_element_id',
    'destination_element_id', 'direction', 'evidence_refs', 'confidence', 'reason'})
_MODEL_ELEMENT_ACTIONS = frozenset({'tap_semantic', 'dismiss_overlay', 'input_verified_text', 'press_enter',
    'clear_verified_text', 'double_tap', 'long_press'})

_KEYBOARD_REQUIRED_FIELDS = frozenset({"visible", "bounds", "layout", "input_mode", "mode_switch"})
_KEYBOARD_OPTIONAL_FIELDS = frozenset({'qwerty_anchors', 'backspace_key', 'enter_key', 'case_mode', 'case_switch',
    'literal_keys', 'layout_switches'})
_INPUT_AUDIT_FIELDS = frozenset({'structure_id', 'bounds', 'fully_visible', 'text', 'placeholder',
    'visible_editable_cues', 'caret_line_index', 'confidence', 'right_button'})


@dataclass(frozen=True)
class _AuditedKeyboard:
    payload: dict[str, Any]
    visible: bool
    layout: str
    input_mode: str
    case_mode: str
    bounds: NormalizedBounds | None
    application_bounds: NormalizedBounds | None
    raw_qwerty_anchors: dict[str, Any] | None
    snapped_qwerty_anchors: dict[str, list[int]] | None
    controls_revoked: bool


@dataclass(frozen=True)
class _AuditedInputControls:
    step: Any = None
    candidate: dict[str, Any] | None = None
    preedit: str = ""
    qwerty: dict[str, Any] | None = None
    backspace: dict[str, Any] | None = None
    clearable_preedit: str = ""
    next_field: dict[str, Any] | None = None
    required_mode: str | None = None
    switch_is_goal: bool = False
    mode: dict[str, Any] | None = None
    literal: dict[str, Any] | None = None
    enter: dict[str, Any] | None = None
    layout: dict[str, Any] | None = None
    case: dict[str, Any] | None = None


def _has_exact_passive_scene_element_fields(value: Any) -> bool:
    return isinstance(value, dict) and frozenset(value) == _PASSIVE_SCENE_ELEMENT_FIELDS


def _contains_action_like_wire_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any((str(key).strip().casefold() in _ACTION_LIKE_WIRE_KEYS
            or _contains_action_like_wire_key(part) for key, part in value.items()))
    if isinstance(value, list):
        return any(_contains_action_like_wire_key(part) for part in value)
    return False


def _is_passive_scene_element_wire_object(value: Any) -> bool:
    return _has_exact_passive_scene_element_fields(value) and not _contains_action_like_wire_key(value)


STAGE_LABELS = {'idle': '空闲', 'checking_stability': '检查画面稳定性', 'waiting_single_step_observation': '等待千问单步完整观察',
    'parsing_single_step_observation': '解析单步完整观察', 'completed': '观察完成', 'failed': '观察安全停止'}


class _SingleStepObserverBase:
    """Shared state and transport for the sole production scene observer."""

    def __init__(self, provider: Any, *, input_lineage_store: TypedInputLineageStorePort | None=None,
        qwerty_row_snapper: Callable[[list[Image.Image] | tuple[Image.Image, ...], dict[str, Any]], dict[str,
        list[int]] | None] | None=None) -> None:
        self.provider = provider
        self.input_lineage_store = input_lineage_store
        self.qwerty_row_snapper = qwerty_row_snapper
        self.last_raw_response = ""
        self.last_diagnostics: dict[str, Any] = {}
        self.last_model_decision: dict[str, Any] | None = None
        self.last_model_decision_fingerprint = ""
        self._stage_lock = threading.RLock()
        self._current_stage = "idle"
        self._last_stage = "idle"
        self.supports_post_action_visual_context = True
        self.supports_runtime_action_contract = True
        self._observation_cache_lock = threading.RLock()
        self._observation_cache: OrderedDict[str, tuple[UIScene, dict[str, Any]]] = OrderedDict()
        self._observation_cache_limit = 32

    def decision_for(self, fingerprint: str) -> dict[str, Any]:
        """Return the action/finish decision emitted with this exact scene response."""

        expected = str(fingerprint or "").strip()
        reject_if(not expected or expected != self.last_model_decision_fingerprint
            or self.last_model_decision is None,
            VisionAgentError("当前可信画面没有同一Qwen响应绑定的action/finish决策。"))
        return dict(self.last_model_decision)

    def _set_stage(self, stage: str) -> None:
        with self._stage_lock:
            self._current_stage = stage
            if stage != 'idle':
                self._last_stage = stage

    def _provider_chat(self, messages: list[dict[str, Any]], *, max_tokens: int, response_format: dict[str,
        str] | None=None) -> str:
        return self.provider._chat(messages, max_tokens=max_tokens, timeout=OBSERVATION_TIMEOUT_SECONDS, max_attempts=1,
            response_format=response_format or {'type': 'json_object'})


class SingleStepGenericSceneObserver(_SingleStepObserverBase):
    """Use one Qwen envelope per fresh scene as the only scene/input observation authority."""

    def status(self) -> dict[str, Any]:
        value = dict(self.provider.status())
        with self._stage_lock:
            current_stage = self._current_stage
            last_stage = self._last_stage
        value.update({'observer_version': SINGLE_STEP_SCENE_OBSERVER_VERSION,
            'scene_protocol': UI_SCENE_PROTOCOL_VERSION,
            'single_step_observation_protocol': SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            'model_role': 'single_step_current_scene_observation', 'supported_app_scope': 'dynamic',
            'hardware_actions_enabled': False, 'current_stage': current_stage,
            'current_stage_label': STAGE_LABELS.get(current_stage, current_stage), 'last_stage': last_stage,
            'last_stage_label': STAGE_LABELS.get(last_stage, last_stage), 'max_online_calls_per_observation': 1,
            'single_step_output_tokens': SINGLE_STEP_OUTPUT_TOKENS,
            'observation_timeout_seconds': OBSERVATION_TIMEOUT_SECONDS,
            'last_scene_enum_values': dict(self.last_diagnostics.get('scene_enum_values') or {}),
            'last_input_structure_shape': dict(self.last_diagnostics.get('input_structure_shape') or {})})
        return value

    def observe(self, *, frames: list[Image.Image], goal_context: dict[str, Any] | None=None,
        device_id: str | None=None, input_lineage_override: TypedInputLineage | None=None,
        post_action_context: post_action_contract.PostActionVisualContext | dict[str, Any] | None=None,
        available_action_kinds: Iterable[str] | None=None) -> UIScene:
        if isinstance(post_action_context, dict):
            post_action_context = post_action_contract.PostActionVisualContext.from_dict(post_action_context)
        elif post_action_context is not None:
            post_action_context.validate()
        self.last_raw_response = ""
        self.last_model_decision = None
        self.last_model_decision_fingerprint = ""
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
            reject_if(len(frames) < 4, VisionAgentError("通用页面观察至少需要4帧。"))
            stability = measure_local_stability(frames, allow_leading_outlier=True)
            reject_if(not stability.stable, VisionAgentError(f"本地多帧稳定性检查未通过：{stability.reason}；不调用模型。"))

            sharpness_scores = [measure_frame_sharpness(item) for item in frames]
            stable_tail_start = max(0, len(frames) - min(3, len(frames)))
            selected_frame_index = max(range(stable_tail_start, len(frames)), key=sharpness_scores.__getitem__)
            frame = frames[selected_frame_index].convert("RGB")
            fingerprint = local_frame_fingerprint(frame)
            context = generic_goal_domain.safe_goal_context(goal_context or {})
            goal = _goal_view(context)
            runtime_actions = _normalize_runtime_action_kinds(available_action_kinds)
            cache_key = _observation_cache_key(device_id=device_id, fingerprint=fingerprint, goal_context=context,
                input_lineage=input_lineage_override, post_action_context=post_action_context,
                available_action_kinds=runtime_actions)
            if cache_key is not None:
                with self._observation_cache_lock:
                    cached_entry = self._observation_cache.get(cache_key)
                    if cached_entry is not None:
                        self._observation_cache.move_to_end(cache_key)
                if cached_entry is not None:
                    cached, cached_decision = cached_entry
                    self.last_model_decision = dict(cached_decision)
                    self.last_model_decision_fingerprint = cached.fingerprint
                    recorder = getattr(self.provider, 'record_observation_cache_hit', None)
                    if callable(recorder):
                        recorder(stage='same_fingerprint_single_step_observation', fingerprint=fingerprint)
                    self.last_diagnostics = {'observer_version': SINGLE_STEP_SCENE_OBSERVER_VERSION,
                        'vision_model': model_identity, 'strategy': 'single_step_exact_fingerprint_cache',
                        'model_calls': 0, 'observation_cache_hit': True,
                        'post_action_visual_context': post_action_context.to_dict() if post_action_context
                        is not None else None, 'fingerprint': fingerprint, 'element_count': len(cached.elements),
                        'elapsed_seconds': round(time.perf_counter() - started, 3)}
                    self._set_stage("completed")
                    return cached

            input_structure_required = goal.input_requested
            active_field_id = goal.field[0]
            verified_lineage: TypedInputLineage | None = None
            if input_lineage_override is not None:
                verified_lineage = input_lineage_override
            ledger_value_hint = verified_lineage.exact_value if verified_lineage is not None else None
            model_frames = tuple(frames[stable_tail_start:]) if input_structure_required else (frame,)

            request_image_sizes = {_image_request_size(item) for item in model_frames}
            reject_if(len(request_image_sizes) != 1, VisionAgentError("同一步发送给Qwen的稳定帧尺寸不一致，不能建立唯一坐标空间。"))
            request_image_size = next(iter(request_image_sizes))

            prompt = _single_step_observation_prompt(context, include_input_structure=input_structure_required,
                current_input_text=ledger_value_hint, image_count=len(model_frames),
                request_image_size=request_image_size, post_action_context=post_action_context,
                available_action_kinds=runtime_actions)
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for (index, item) in enumerate(model_frames, start=1):
                content.extend(({'type': 'text', 'text': f'IMAGE {index} - SAME STABLE PHONE SURFACE'},
                    {'type': 'image_url', 'image_url': {'url': _image_data_url(item.convert('RGB'))}}))
            self._set_stage("waiting_single_step_observation")
            scope_factory = getattr(self.provider, "call_scope", None)
            scope = scope_factory(stage='single_step_observation',
                fingerprint=fingerprint) if callable(scope_factory) else nullcontext()
            call_started = time.perf_counter()
            model_calls = 1
            with scope:
                raw = self._provider_chat([_json_only_system_message(), {'role': 'user', 'content': content}],
                    max_tokens=SINGLE_STEP_OUTPUT_TOKENS, response_format={'type': 'json_object'})
            call_elapsed = round(time.perf_counter() - call_started, 3)
            self.last_raw_response = raw
            self._set_stage("parsing_single_step_observation")
            envelope = _parse_single_step_observation_envelope(raw, input_structure_required=input_structure_required,
                request_image_size=request_image_size)
            model_decision = dict(envelope["decision"])
            scene_payload = dict(envelope["scene"])
            single_step_input_surface = _single_step_input_surface_attestation(scene_payload,
                goal_context=context) if input_structure_required else None
            if input_structure_required:
                _strip_preliminary_input_elements(scene_payload, context)
            obstructions = consensus_top_edge_obstructions(frames[stable_tail_start:])
            scene = _suppress_obscured_input_evidence(_parse_scene(json.dumps(scene_payload, ensure_ascii=False,
                separators=(',', ':')), fingerprint=fingerprint,
                camera_layout_orientation=_camera_layout_orientation(frame)), obstructions, fingerprint=fingerprint)

            if input_structure_required:
                # Trust lineage only after this response establishes foreground App/screen identity.
                if (verified_lineage is not None and (not (isinstance(device_id,
                    str) and verified_lineage.matches_typed_context(device_id=device_id, app_id=scene.app_id,
                    screen_id=scene.screen_id, input_field_id=active_field_id)))):
                    verified_lineage = None
                if (verified_lineage is None and self.input_lineage_store is not None and isinstance(device_id,
                    str) and device_id.strip()):
                    stored = self.input_lineage_store.load(device_id)
                    if (stored is not None and stored.matches_typed_context(device_id=device_id, app_id=scene.app_id,
                        screen_id=scene.screen_id, input_field_id=active_field_id,
                        now_epoch=float(self.input_lineage_store.clock()),
                        ttl_seconds=self.input_lineage_store.ttl_seconds)):
                        verified_lineage = stored
                ledger_value_hint = verified_lineage.exact_value if verified_lineage is not None else None
                input_payload = envelope["input_structure"]
                assert isinstance(input_payload, dict)
                scene = _suppress_obscured_input_evidence(_apply_input_structure_audit(scene, json.dumps(input_payload,
                    ensure_ascii=False, separators=(',', ':')), fingerprint=fingerprint, goal_context=context,
                    verified_input_lineage=verified_lineage, device_id=device_id, lineage_frame=frame,
                    qwerty_row_snapper=self.qwerty_row_snapper, qwerty_row_frames=frames[stable_tail_start:],
                    single_step_input_surface=single_step_input_surface), obstructions, fingerprint=fingerprint)
                reject_if(
                    (goal.transaction_text or goal.target_only) and (not _input_audit_established_local_target(scene))
                    and (not _focus_only_input_surface_established(scene)),
                    VisionAgentError("单步完整观察没有建立当前输入事务的唯一本地目标。"),
                )

            missing_evidence = [item.element_id for item in scene.elements if item.states.get('goal_relevant') is True
                and (not any((value.strip() for value in item.evidence)))]
            reject_if(missing_evidence, VisionAgentError("目标相关元素缺少原始可见证据，不能建立可信候选：" + ",".join(missing_evidence)))
            target = scene.unique_trusted_goal_element()
            completion = scene.trusted_completion_evidence()
            reject_if(
                not scene.stable or (float(scene.confidence) < MIN_TARGET_CONFIDENCE and target is None
                and (not completion)),
                VisionAgentError("页面不稳定或整体置信度不足，不能建立可信候选。"),
            )

            self.last_model_decision = model_decision
            self.last_model_decision_fingerprint = scene.fingerprint

            self.last_diagnostics = {'observer_version': SINGLE_STEP_SCENE_OBSERVER_VERSION,
                'vision_model': public_model_identity(self.provider.status()),
                'strategy': 'single_step_current_scene_observation',
                'protocol_version': SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION, 'model_calls': 1,
                'online_stages': ['single_step_observation'],
                'input_structure_in_same_response': input_structure_required, 'remote_retry_used': False,
                'observation_cache_hit': False, 'post_action_visual_context': post_action_context.to_dict(
                ) if post_action_context is not None else None, 'selected_frame_index': selected_frame_index,
                'stable_tail_start_index': stable_tail_start, 'local_stability': stability.to_dict(),
                'frame_sharpness_scores': [round(value, 3) for value in sharpness_scores],
                'frame_size': list(frame.size), 'request_image_size': list(request_image_size),
                'coordinate_normalization': envelope.get('coordinate_normalization'), 'fingerprint': fingerprint,
                'element_count': len(scene.elements), 'decision_status': model_decision['status'],
                'available_action_kinds': list(runtime_actions),
                'model_call_elapsed_seconds': [call_elapsed],
                'model_call_token_budgets': [SINGLE_STEP_OUTPUT_TOKENS],
                'elapsed_seconds': round(time.perf_counter() - started, 3)}
            if cache_key is not None:
                with self._observation_cache_lock:
                    self._observation_cache[cache_key] = (scene, model_decision)
                    self._observation_cache.move_to_end(cache_key)
                    while len(self._observation_cache) > self._observation_cache_limit:
                        self._observation_cache.popitem(last=False)
            self._set_stage("completed")
            return scene
        except Exception as exc:
            failed_stage = self.status()["last_stage"]
            self._set_stage("failed")
            self.last_diagnostics = {'observer_version': SINGLE_STEP_SCENE_OBSERVER_VERSION,
                'vision_model': public_model_identity(self.provider.status()),
                'strategy': 'single_step_current_scene_observation',
                'protocol_version': SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION, 'model_calls': model_calls,
                'online_stages': ['single_step_observation'] if model_calls else [],
                'input_structure_in_same_response': input_structure_required, 'remote_retry_used': False,
                'failed_stage': failed_stage, 'post_action_visual_context': post_action_context.to_dict(
                ) if post_action_context is not None else None, 'fingerprint': fingerprint, 'error': str(exc),
                'error_type': classify_qwen_error(exc, raw_response=self.last_raw_response),
                'safe_stop_reason': '单次模型输出未建立完整可信观察；没有发起第二次Qwen请求，控制器与机械臂均未执行。',
                'raw_response_length': len(self.last_raw_response),
                'raw_response_excerpt': self.last_raw_response[:1000],
                'elapsed_seconds': round(time.perf_counter() - started, 3)}
            raise
        finally:
            self._set_stage("idle")


def _horizontal_overlap_ratio(first: NormalizedBounds, second: tuple[int, int, int, int]) -> float:
    overlap = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    return overlap / max(1.0, first[2] - first[0])


def _input_evidence_is_obscured(bounds: NormalizedBounds, obstruction: VisualObstruction) -> bool:
    if obstruction.kind != 'top_edge_opaque_band':
        return False
    horizontal_overlap = _horizontal_overlap_ratio(bounds, obstruction.bounds)
    vertical_gap = bounds[1] - obstruction.bounds[3]
    obstruction_height = obstruction.bounds[3] - obstruction.bounds[1]
    return horizontal_overlap >= 0.15 and vertical_gap <= max(20.0, obstruction_height * 0.5)


def _suppress_obscured_input_evidence(scene: UIScene, obstructions: tuple[VisualObstruction, ...], *,
    fingerprint: str) -> UIScene:
    """Prevent a crop/model claim from restoring pixels hidden in the full frame."""

    if not obstructions:
        return scene
    value = scene.to_dict()
    changed = False
    for element in value.get('elements') or []:
        if not isinstance(element, dict) or element.get('role') != 'input':
            continue
        raw_bounds = element.get("bounds")
        if not isinstance(raw_bounds, list) or len(raw_bounds) != 4:
            continue
        bounds = tuple(float(part) * 1000 for part in raw_bounds)
        matched = next((obstruction for obstruction in obstructions if _input_evidence_is_obscured(bounds,
            obstruction)), None)
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
    return UIScene.from_dict(value, coordinate_scale=1.0, stable_override=True, fingerprint_override=fingerprint)


def _json_only_system_message() -> dict[str, str]:
    return {'role': 'system', 'content': '你是只读页面观察器。只输出一个语法完整的JSON对象；禁止Markdown、解释、思考过程、代码围栏、JSON字符串套壳或对象前后的任何文字。'}


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

def _single_step_observation_prompt(context: dict[str, Any], *, include_input_structure: bool,
    current_input_text: str | None, image_count: int, request_image_size: tuple[int, int],
    post_action_context: post_action_contract.PostActionVisualContext | None,
    available_action_kinds: tuple[str, ...]) -> str:
    request_width, request_height = request_image_size
    scene_contract = _compact_prompt(context, wire_height=request_height,
        input_structure_is_value_authority=include_input_structure)
    if include_input_structure:
        input_contract = _input_structure_audit_prompt(context, current_input_text=current_input_text,
            wire_height=request_height)
        input_rule = "input_structure必须是完整输入结构对象，使用下面INPUT CONTRACT的字段和值规则；不得为null。"
    else:
        input_contract = "本轮子目标与文字输入无关。"
        input_rule = "input_structure必须为null，不得额外枚举键盘或输入结构。"
    temporal_rule = (f"共有{image_count}张同一稳定手机画面的时间对齐帧。只把它们合并为一个当前状态；"
        "闪烁光标可从任一帧读取，其他瞬态不得合并。" if image_count > 1 else "只有一张当前稳定手机画面。")
    if post_action_context is None:
        post_action_rule = "本轮不是本地已签发的动作后观察；不得猜测此前执行过任何动作。"
    else:
        payload = json.dumps(post_action_context.to_dict(), ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        post_action_rule = ("本轮是一个物理动作后的首次观察。本地只读typed摘要如下：\n" + payload
            + "\n它只证明该canonical动作已到达物理执行层，并说明需要核对的typed后置条件；\n"
            "outcome=pending_visual_verification，绝不等于matched，也不能迫使你把预期写成事实。\n"
            "当前JPEG仍是当前画面的唯一权威：符合时报告可见结果，不符合时如实报告矛盾。\n"
            "识别时先判断最外层系统/App表面，再判断其中嵌入的卡片、预览或子内容；嵌入内容\n"
            "所属App不能替代承载它的最外层系统表面。该摘要只帮助选择核对重点，不授权动作。\n")
    return _render_prompt("single_step_observation.txt", SCENE_CONTRACT=scene_contract,
        INPUT_CONTRACT=input_contract, TEMPORAL_RULE=temporal_rule, POST_ACTION_RULE=post_action_rule,
        INPUT_RULE=input_rule, REQUEST_WIDTH=str(request_width), REQUEST_HEIGHT=str(request_height),
        OBSERVATION_PROTOCOL=SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION, SCENE_PROTOCOL=UI_SCENE_PROTOCOL_VERSION,
        AVAILABLE_ACTIONS_JSON=json.dumps(list(available_action_kinds), ensure_ascii=False, separators=(',', ':')))


def _normalize_runtime_action_kinds(value: Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return tuple(sorted(CANONICAL_ACTION_KINDS))
    try:
        normalized = frozenset(str(item or '').strip() for item in value)
    except TypeError as exc:
        raise VisionAgentError("本轮可用动作集合必须是可迭代字符串。") from exc
    reject_if(not normalized or '' in normalized or normalized - CANONICAL_ACTION_KINDS,
        VisionAgentError("本轮可用动作集合为空或包含协议外动作。"))
    return tuple(sorted(normalized))


def _normalize_single_step_wire_coordinates(payload: dict[str, Any], *, request_image_size: tuple[int,
    int]) -> dict[str, Any]:
    """Convert the sole request-bound axis grid to canonical 0..1000."""
    coordinate_space = payload.get("coordinate_space")
    request_width, request_height = request_image_size
    expected = {"kind": "axis_grid", "width": 1000, "height": request_height}
    reject_if(coordinate_space != expected, UISceneError("单步观察必须声明唯一coordinate_space。"))
    reject_if(request_width <= 0 or request_height <= 0, UISceneError("本轮Qwen请求图片尺寸无效。"))

    def numbers(value: Any, length: int, label: str) -> list[float]:
        reject_if(
            not isinstance(value, (list, tuple)) or len(value) != length or any((isinstance(part,
            bool) or not isinstance(part, (int, float)) for part in value)),
            UISceneError(f"coordinate_space内存在无效{label}。"),
        )
        return [float(part) for part in value]

    def normalize(node: Any, *, anchors: bool=False) -> None:
        if isinstance(node, list):
            for item in node:
                normalize(item, anchors=anchors)
            return
        if not isinstance(node, dict):
            return
        for (key, value) in tuple(node.items()):
            if key == 'bounds' and value is not None:
                left, top, right, bottom = numbers(value, 4, "bounds")
                reject_if(not (0 <= left < right <= 1000 and 0 <= top < bottom <= request_height), UISceneError("bounds超出声明的axis_grid。"))
                node[key] = [round(left), round(top * 1000 / request_height), round(right),
                    round(bottom * 1000 / request_height)]
                reject_if(not _valid_1000_bounds(node[key]), UISceneError("axis_grid换算后bounds退化。"))
            elif anchors and value is not None:
                x, y = numbers(value, 2, "锚点")
                reject_if(not (0 <= x <= 1000 and 0 <= y <= request_height), UISceneError("锚点超出声明的axis_grid。"))
                node[key] = [round(x), round(y * 1000 / request_height)]
            else:
                normalize(value, anchors=(key == "qwerty_anchors"))

    normalize(payload.get("scene"))
    normalize(payload.get("input_structure"))
    payload.pop("coordinate_space", None)
    return {'wire_kind': 'axis_grid', 'wire_extent': [1000, request_height], 'request_image_size': [request_width,
        request_height], 'canonical_extent': [1000, 1000], 'applied': True}


def _parse_single_step_observation_envelope(raw: str, *, input_structure_required: bool, request_image_size: tuple[int,
    int]) -> dict[str, Any]:
    """Parse one current-scene response without remote repair or resampling."""

    try:
        payload = _extract_json_object(raw, reject_duplicate_keys=True)
        required = {'protocol_version', 'coordinate_space', 'scene', 'input_structure', 'decision'}
        if set(payload) != required:
            missing = sorted(required - set(payload))
            extra = sorted(set(payload) - required)
            details = []
            if missing:
                details.append("缺少字段：" + ", ".join(missing))
            if extra:
                details.append("包含协议外字段：" + ", ".join(extra))
            raise UISceneError("单步观察封装结构无效；" + "；".join(details))
        reject_if(payload['protocol_version'] != SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION, UISceneError("单步观察协议版本不匹配。"))
        reject_if(not isinstance(payload['scene'], dict), UISceneError("单步观察scene必须是对象。"))
        coordinate_normalization = _normalize_single_step_wire_coordinates(payload,
            request_image_size=request_image_size)
        input_payload = payload["input_structure"]
        reject_if(input_structure_required and (not isinstance(input_payload, dict)), UISceneError("输入子目标必须在同一响应返回input_structure对象。"))
        reject_if(not input_structure_required and input_payload is not None, UISceneError("非输入子目标的input_structure必须为null。"))
        decision = _parse_model_step_decision(payload['decision'])
        return {'scene': payload['scene'], 'input_structure': payload['input_structure'],
            'decision': decision, 'coordinate_normalization': coordinate_normalization}
    except (UISceneError, ValueError, TypeError) as exc:
        raise VisionAgentError(f"单步完整观察结果不符合协议：{exc}") from exc


def _parse_model_step_decision(value: Any) -> dict[str, Any]:
    """Validate the one action/finish choice emitted with the current scene."""

    reject_if(not isinstance(value, dict) or set(value) != _MODEL_DECISION_FIELDS,
        UISceneError("decision字段不完整或包含协议外字段。"))
    status = value.get('status')
    reject_if(status not in {'action', 'finish'}, UISceneError("decision.status只允许action或finish。"))
    confidence = value.get('confidence')
    reject_if(isinstance(confidence, bool) or not isinstance(confidence, (int, float))
        or not 0.0 <= float(confidence) <= 1.0, UISceneError("decision.confidence无效。"))
    reason = value.get('reason')
    reject_if(not isinstance(reason, str) or not reason.strip(), UISceneError("decision.reason不能为空。"))
    refs = value.get('evidence_refs')
    reject_if(not isinstance(refs, list) or len(refs) > 8
        or any(not isinstance(item, str) or not item.strip() for item in refs)
        or len(refs) != len(set(refs)), UISceneError("decision.evidence_refs无效。"))

    text_fields = ('action', 'element_id', 'source_element_id', 'destination_element_id', 'direction')
    for name in text_fields:
        part = value.get(name)
        reject_if(part is not None and (not isinstance(part, str) or not part.strip()),
            UISceneError(f"decision.{name}必须为非空字符串或null。"))

    action = value.get('action')
    element_id = value.get('element_id')
    source_id = value.get('source_element_id')
    destination_id = value.get('destination_element_id')
    direction = value.get('direction')
    if status == 'finish':
        reject_if(any(item is not None for item in (action, element_id, source_id, destination_id, direction)),
            UISceneError("finish不得携带动作引用。"))
        reject_if(not refs, UISceneError("finish必须引用同一scene中的可见证据。"))
    else:
        reject_if(action not in CANONICAL_ACTION_KINDS, UISceneError("decision.action不在canonical动作集合。"))
        reject_if(refs, UISceneError("action决策不得携带完成证据。"))
        if action in _MODEL_ELEMENT_ACTIONS:
            reject_if(element_id is None or any(item is not None for item in (source_id, destination_id, direction)),
                UISceneError("元素动作必须且只能引用一个element_id。"))
        elif action == 'drag':
            reject_if(element_id is not None or direction is not None or source_id is None or destination_id is None
                or source_id == destination_id, UISceneError("drag必须且只能引用不同的起点和终点元素。"))
        elif action == 'swipe':
            reject_if(source_id is not None or destination_id is not None
                or direction not in {'up', 'down', 'left', 'right'}, UISceneError("swipe必须声明唯一方向。"))
        else:
            reject_if(any(item is not None for item in (element_id, source_id, destination_id, direction)),
                UISceneError("系统或容器动作不得携带元素或方向字段。"))
    return {**value, 'reason': reason.strip()[:500], 'confidence': float(confidence),
        'evidence_refs': list(refs)}


def _compact_prompt(context: dict[str, Any], *, wire_height: int=1000,
    input_structure_is_value_authority: bool=False) -> str:
    context = _goal_view(context).observation_context
    if _goal_view(context).mode_switch_requested:
        keyboard_switch_rule = (" 当前子目标明确要求切换键盘输入模式；本轮快速观察不得在elements中报告或定位"
            "任何模式切换键。后续独立全帧输入结构审计是模式、方向和模式键几何的唯一权威。"
            "普通输入框和键盘可见事实仍可报告，但不得据此建议动作。")
    else:
        keyboard_switch_rule = KEYBOARD_MODE_SWITCH_OBSERVATION_RULE
    input_rule = (("本轮input_structure.application_inputs.text是应用输入正文空/非空事实的"
        "唯一视觉权威。scene中的role=input只可报告一个目标相关输入表面的身份、可见边界、"
        "goal_relevant、fully_visible和外观证据；不得在scene.states.value中重复正文，也不得用scene正文"
        "否决input_structure。键盘模式、IME和可执行输入几何只在同一响应的input_structure中报告。"
        + LOCAL_TEXT_CLEAR_OBSERVATION_RULE) if input_structure_is_value_authority else
        INPUT_VALUE_AND_MODE_OBSERVATION_RULE + keyboard_switch_rule + LOCAL_TEXT_CLEAR_OBSERVATION_RULE)
    return _render_prompt("compact_scene.txt", CONTEXT=json.dumps(context, ensure_ascii=False, separators=(',', ':')),
        INPUT_RULE=input_rule, MAX_ELEMENTS=str(MAX_COMPACT_ELEMENTS), WIRE_HEIGHT=str(wire_height),
        SCENE_PROTOCOL=UI_SCENE_PROTOCOL_VERSION)


def _input_audit_literal_key_targets(context: dict[str, Any], *, current_input_text: str | None=None) -> tuple[str,
    ...]:
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
            if next_step.kind == 'literal_key' and next_step.segment not in {'\r', '\n'}:
                return (next_step.segment,)
            return ()
    targets: list[str] = []
    for character in text:
        if character in {'\r', '\n', '\t'} or character.isalpha():
            continue
        if character not in targets:
            targets.append(character)
        if len(targets) == 8:
            break
    return tuple(targets)


def _input_structure_audit_prompt(context: dict[str, Any], *, current_input_text: str | None=None,
    wire_height: int=1000) -> str:
    context = _goal_view(context).observation_context
    goal = _goal_view(context)
    keyboard_min_height = max(1, round(180 * wire_height / 1000))
    literal_targets = _input_audit_literal_key_targets(context, current_input_text=current_input_text)
    literal_example = []
    if literal_targets:
        value = literal_targets[0]
        literal_example = [{'value': value, 'label': 'Space' if value == ' ' else value,
            'key_kind': 'space' if value == ' ' else 'character', 'bounds': [0, 0, 1000, wire_height],
            'confidence': 0.0, 'fully_visible': True}]
    field_id, field_label, multiline = goal.field
    target_text = goal.transaction_text or goal.explicit_text
    enter_required = False
    if target_text and current_input_text is not None and multiline:
        try:
            step = plan_next_verified_input(target_text, current_input_text)
        except (ValueError, VerifiedTextTransactionError):
            step = None
        enter_required = bool(step is not None and step.kind == 'literal_key' and step.segment == '\n')
    return _render_prompt("input_structure_audit.txt",
        CONTEXT=json.dumps(context, ensure_ascii=False, separators=(',', ':')),
        FIELD_ID=json.dumps(field_id, ensure_ascii=False), FIELD_LABEL=json.dumps(field_label, ensure_ascii=False),
        TARGET_ONLY_CLEAR=str(goal.target_only).lower(), ENTER_REQUIRED=str(enter_required).lower(),
        MULTILINE=str(multiline).lower(), LITERAL_TARGETS=json.dumps(literal_targets, ensure_ascii=False,
        separators=(',', ':')), LITERAL_KEYS_EXAMPLE=json.dumps(literal_example, ensure_ascii=False,
        separators=(',', ':')), WIRE_HEIGHT=str(wire_height), KEYBOARD_MIN_HEIGHT=str(keyboard_min_height),
        AUDIT_VERSION=INPUT_STRUCTURE_AUDIT_VERSION)


def _parse_scene(raw: str, *, fingerprint: str, camera_layout_orientation: str | None=None) -> UIScene:
    try:
        payload = _extract_json_object(raw)
        reject_if('system_ui' not in payload, UISceneError("新观察必须显式返回 scene.system_ui；无法判断时两项都写 unknown。"))
        _drop_forbidden_camera_alignment_evidence(payload)
        reject_if('camera_alignment' not in payload, UISceneError("新观察必须显式返回 scene.camera_alignment。"))
        alignment = CameraAlignmentFacts.from_dict(payload["camera_alignment"])
        reject_if(camera_layout_orientation is not None and alignment.camera_layout_orientation != camera_layout_orientation, UISceneError("模型报告的相机画布方向与本地稳定帧尺寸不一致。"))
        _strip_model_authored_local_attestations(payload)
        return UIScene.from_dict(payload, coordinate_scale=1000.0, stable_override=True,
            fingerprint_override=fingerprint)
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
    if not isinstance(evidence, list) or not all((isinstance(item, str) for item in evidence)):
        return
    retained = [item for item in evidence if camera_alignment_evidence_is_safe(item)]
    if retained and len(retained) != len(evidence):
        alignment["evidence"] = retained
        return
    if not evidence or retained:
        return
    required = {'camera_layout_orientation', 'phone_content_rotation', 'confidence', 'evidence'}
    confidence = alignment.get("confidence")
    if (
        set(alignment) != required
        or alignment.get("camera_layout_orientation") not in CAMERA_LAYOUT_ORIENTATIONS
        or alignment.get("phone_content_rotation") not in PHONE_CONTENT_ROTATIONS
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        # The repair never hides malformed scalars; the strict parser still rejects them.
        return
    # HUD/coordinate text cannot prove rotation; revoke only that fact and retain the independent action gate.
    alignment["phone_content_rotation"] = "unknown"
    alignment["confidence"] = 0.0
    alignment["evidence"] = []


def _goal_view(context: dict[str, Any]) -> generic_goal_domain.ActiveVisualGoal:
    return generic_goal_domain.ActiveVisualGoal.from_context(context)


def _valid_1000_bounds(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    if not all((isinstance(part, (int, float)) and (not isinstance(part, bool)) for part in value)):
        return False
    left, top, right, bottom = (float(part) for part in value)
    return 0 <= left < right <= 1000 and 0 <= top < bottom <= 1000


def _strip_preliminary_input_elements(payload: dict[str, Any], goal_context: dict[str, Any]) -> bool:
    goal = _goal_view(goal_context)
    if not goal.input_requested:
        return False
    elements = payload.get("elements")
    if not isinstance(elements, list):
        return False
    mode_only = goal.mode_switch_requested
    text_input = goal.has_explicit_text
    keyboard_meanings = {'input_exact_enter_key', 'input_exact_literal_key', 'switch_keyboard_case',
        'switch_keyboard_input_mode', 'switch_keyboard_layout'}
    retained: list[Any] = []
    removed = False
    for item in elements:
        passive = _is_passive_scene_element_wire_object(item)
        role = str(item.get("role") or "").strip() if passive else ""
        meaning = str(item.get("meaning") or "").strip().casefold() if passive else ""
        dedicated = bool(mode_only or role == 'input' or (text_input and (role == 'container' and 'keyboard' in meaning
            or (role in {'button', 'key'} and meaning in keyboard_meanings))))
        if passive and dedicated:
            removed = True
        else:
            retained.append(item)
    if removed:
        payload["elements"] = retained
    return removed


def _single_step_input_surface_attestation(payload: dict[str, Any], *, goal_context: dict[str, Any]) -> dict[str,
    Any] | None:
    """Keep one same-envelope input-surface fact without text or executable geometry authority."""

    goal = _goal_view(goal_context)
    if not (goal.transaction_text or goal.target_only):
        return None
    elements = payload.get("elements")
    if not isinstance(elements, list):
        return None
    candidates: list[dict[str, Any]] = []
    for item in elements:
        if not _has_exact_passive_scene_element_fields(item):
            continue
        states = item.get("states")
        evidence = item.get("evidence")
        confidence = item.get("confidence")
        raw_bounds = item.get("bounds")
        if (str(item.get('role') or '').strip() != 'input' or not isinstance(item.get('element_id'),
            str) or (not item['element_id'].strip()) or str(item['element_id']).startswith('local_audited_')
            or (not isinstance(item.get('meaning'), str)) or (not item['meaning'].strip())
            or (not isinstance(item.get('label'), str)) or (not isinstance(states,
            dict)) or (states.get('goal_relevant') is not True) or (states.get('fully_visible') is not True)
            or isinstance(confidence, bool) or (not isinstance(confidence, (int,
            float))) or (float(confidence) < 0.9) or (not isinstance(evidence,
            list)) or (not evidence) or any((not isinstance(value,
            str) or not value.strip() for value in evidence)) or (not _valid_1000_bounds(raw_bounds))):
            continue
        bounds = [float(value) for value in raw_bounds]
        if max(bounds) <= 1.0:
            bounds = [value * 1000.0 for value in bounds]
        candidates.append({'element_id': item['element_id'].strip(), 'meaning': item['meaning'].strip(),
            'label': item['label'].strip()[:200], 'bounds': bounds, 'confidence': float(confidence),
            'evidence': tuple((str(value).strip()[:200] for value in evidence))})
    return candidates[0] if len(candidates) == 1 else None


def _input_audit_established_local_target(scene: UIScene) -> bool:
    """Return true only for authority minted by the dedicated input audit."""

    local_ids = {'local_audited_input_1', 'local_audited_keyboard_mode_switch_1', 'local_audited_ime_candidate_1',
        'local_audited_literal_key_1', 'local_audited_enter_key_1', 'local_audited_next_field_key_1',
        'local_audited_keyboard_layout_switch_1', 'local_audited_keyboard_case_switch_1'}
    candidates = tuple((element for element in scene.elements if element.element_id in local_ids
        and element.states.get('goal_relevant') is True and (element.states.get('fully_visible') is True)
        and (float(element.confidence) >= MIN_TARGET_CONFIDENCE)))
    return len(candidates) == 1


def _focus_only_input_surface_established(scene: UIScene) -> bool:
    """Accept one sanitized focus-only surface with no text, field, or keyboard authority."""

    candidates = tuple((element for element in scene.elements if element.role == 'input'
        and element.states.get('focus_only_input_surface') is True and (element.states.get('goal_relevant') is True)
        and (element.states.get('fully_visible') is True) and (float(element.confidence) >= MIN_TARGET_CONFIDENCE)
        and any((str(item).strip() for item in element.evidence))))
    if len(candidates) != 1:
        return False
    unique = scene.unique_trusted_goal_element()
    return unique is not None and unique.element_id == candidates[0].element_id


def _focus_only_compact_input_surface(attestation: dict[str, Any] | None, *, keyboard_visible: bool) -> dict[str,
    Any] | None:
    """Sanitize one compact input into focus-only authority without copying its transcription."""

    if keyboard_visible or not isinstance(attestation, dict):
        return None
    if set(attestation) != {'element_id', 'meaning', 'label', 'bounds', 'confidence', 'evidence'}:
        return None
    element_id = str(attestation.get("element_id") or "").strip()
    meaning = str(attestation.get("meaning") or "").strip()
    label = str(attestation.get("label") or "").strip()[:200]
    bounds = attestation.get("bounds")
    evidence = attestation.get("evidence")
    confidence = attestation.get("confidence")
    if (not element_id or element_id.startswith('local_audited_') or (not meaning) or (not isinstance(bounds, (list,
        tuple))) or (len(bounds) != 4) or any((isinstance(value, bool) or not isinstance(value, (int,
        float)) or (not 0 <= float(value) <= 1000) for value in bounds)) or isinstance(confidence,
        bool) or (not isinstance(confidence, (int, float))) or (float(confidence) < MIN_TARGET_CONFIDENCE)
        or (not isinstance(evidence, (list, tuple))) or (not evidence) or any((not isinstance(item,
        str) or not item.strip() for item in evidence))):
        return None
    return {'element_id': element_id, 'role': 'input', 'meaning': meaning,
        'bounds': [float(value) / 1000.0 for value in bounds], 'confidence': float(confidence), 'label': label,
        'states': {'goal_relevant': True, 'fully_visible': True, 'focus_only_input_surface': True},
        'evidence': [str(item).strip()[:200] for item in evidence]}


def _adjacent_exact_preedit_cue(trusted_input: dict[str, Any], trusted_preedits: list[dict[str, Any]],
    exact_text: str) -> bool:
    """Accept one exact non-authoritative cue beside the same input surface."""

    if (not exact_text or len(trusted_preedits) != 1 or trusted_preedits[0].get('text') != exact_text
        or trusted_preedits[0].get('candidates')):
        return False
    input_box = tuple(float(value) for value in trusted_input["input_bounds"])
    preedit_box = tuple(float(value) for value in trusted_preedits[0]["bounds"])
    return _horizontally_adjacent(input_box, preedit_box)


def _horizontally_adjacent(first: tuple[float, ...], second: tuple[float, ...]) -> bool:
    overlap = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    smaller_width = min(first[2] - first[0], second[2] - second[0])
    vertical_gap = max(0.0, second[1] - first[3], first[1] - second[3])
    return bool(smaller_width > 0 and overlap / smaller_width >= 0.6 and vertical_gap <= 100)


def _resolve_pending_ime_candidate_input_state(trusted_input: dict[str, Any], trusted_preedits: list[dict[str, Any]], *,
    application_input_count: int, verified_input_lineage: TypedInputLineage | None, device_id: str, app_id: str,
    screen_id: str, input_field_id: str, authorized_text: str, keyboard_input_mode: str) -> dict[str, Any] | None:
    """Resolve one receipt-bound IME candidate using only lineage and the dedicated input audit."""

    if (application_input_count != 1 or verified_input_lineage is None
        or verified_input_lineage.source != 'pending_verified_ime_candidate_action' or (keyboard_input_mode not
        in {'direct_latin', 'chinese_pinyin'}) or (not input_field_id) or (input_field_id == 'unknown')
        or (input_field_id != verified_input_lineage.input_field_id) or (not isinstance(authorized_text,
        str)) or (not authorized_text.startswith(verified_input_lineage.exact_value))
        or (not isinstance(trusted_input.get('caret_line_index'), int)) or (len(trusted_preedits) != 1)):
        return None
    exact_value = verified_input_lineage.exact_value
    raw_text = trusted_input.get("text")
    if not isinstance(raw_text, str) or raw_text not in {'', exact_value}:
        return None
    preedit = trusted_preedits[0]
    preedit_text = preedit.get("text")
    candidates = preedit.get("candidates")
    if (not isinstance(preedit_text, str) or not preedit_text or '\r' in preedit_text or ('\n' in preedit_text)
        or (not exact_value.endswith(preedit_text)) or (not isinstance(candidates, list)) or (sum((isinstance(candidate,
        dict) and candidate.get('text') == preedit_text for candidate in candidates)) != 1)):
        return None
    input_box = tuple(float(value) for value in trusted_input["input_bounds"])
    preedit_box = tuple(float(value) for value in preedit["bounds"])
    if _bounds_overlap_ratio(input_box, preedit_box) < 0.9 or _bounds_overlap_ratio(preedit_box, input_box) < 0.9:
        return None
    normalized_bounds = tuple(value / 1000.0 for value in input_box)
    if (not verified_input_lineage.matches_pending_input_state_surface(device_id=device_id, app_id=app_id,
        screen_id=screen_id, input_bounds=normalized_bounds, input_field_id=input_field_id)):
        return None
    return {'committed_value': exact_value, 'consumed_preedit': preedit}


def _authorized_exact_committed_prefix_cue(trusted_input: dict[str, Any], trusted_preedits: list[dict[str, Any]], *,
    goal: generic_goal_domain.ActiveVisualGoal, keyboard_input_mode: str) -> str:
    """Recover one committed authorized prefix inside the typed field."""

    field_id, _field_label, _multiline = goal.field
    authorized = goal.transaction_text
    cues = trusted_input.get("visible_editable_cues")
    if (not field_id or field_id == 'unknown' or (not isinstance(authorized,
        str)) or (not authorized) or (keyboard_input_mode != 'direct_latin') or (trusted_input.get('text') != '')
        or (not isinstance(cues, list))):
        return ""
    non_text_cues = {'border', 'caret', 'cursor', 'focus border', 'focus ring', 'outline'}
    literal_cues = tuple((item.strip() for item in cues if isinstance(item,
        str) and item.strip() and (item.strip().casefold() not in non_text_cues)))
    if len(literal_cues) != 1:
        return ""
    cue = literal_cues[0]
    if (not authorized.startswith(cue) or cue == trusted_input.get('placeholder') or cue
        in trusted_input.get('field_labels', ()) or ('\r' in cue) or ('\n' in cue)):
        return ""
    if trusted_preedits and (not _adjacent_exact_preedit_cue(trusted_input, trusted_preedits, cue)):
        return ""
    return cue


def _clear_goal_unique_committed_cue(trusted_input: dict[str, Any], trusted_preedits: list[dict[str, Any]], *,
    keyboard_input_mode: str) -> tuple[str, int]:
    """Recover committed glyphs for clear-all while treating extra visual rows only as delete units."""

    cues = trusted_input.get("visible_editable_cues")
    if (keyboard_input_mode != 'direct_latin' or trusted_input.get('text') != '' or trusted_preedits
        or (not isinstance(cues, list))):
        return "", 0
    decorative = {'border', 'caret', 'cursor', 'focus border', 'focus ring', 'outline', '|'}
    literals = tuple((item for item in cues if isinstance(item,
        str) and item.strip() and (item.strip().casefold() not in decorative) and ('caret' not in item.casefold())
        and ('cursor' not in item.casefold()) and ('光标' not in item) and ('插入符' not in item)))
    if len(literals) != 1:
        return "", 0
    cue = literals[0]
    if (cue == trusted_input.get('placeholder') or cue in trusted_input.get('field_labels',
        ()) or '\r' in cue or ('\n' in cue)):
        return "", 0
    caret_line_index = trusted_input.get("caret_line_index")
    extra_rows = caret_line_index if isinstance(caret_line_index, int) and (not isinstance(caret_line_index,
        bool)) and (caret_line_index > 0) else 0
    return cue, extra_rows


def _locally_detect_caret_visual_row(trusted_input: dict[str, Any], *, frames: tuple[Image.Image, ...] | None,
    qwerty_anchors: dict[str, list[int]] | None, allow_goal_bound_local_detection: bool=False) -> int | None:
    """Locate a caret row only within the audited field band above calibrated QWERTY rows."""

    cues = trusted_input.get("visible_editable_cues")
    model_attests_caret = isinstance(cues, list) and any((isinstance(cue,
        str) and (cue.strip() == '|' or 'caret' in cue.casefold() or 'cursor' in cue.casefold() or ('光标' in cue)
        or ('插入符' in cue)) for cue in cues))
    if not model_attests_caret and (not allow_goal_bound_local_detection):
        return None
    if (not frames or not isinstance(qwerty_anchors, dict) or (not all((key in qwerty_anchors for key in ('q', 'p', 'a',
        'l'))))):
        return None
    input_bounds = trusted_input.get("input_bounds")
    if not _valid_1000_bounds(input_bounds):
        return None
    q_points = (qwerty_anchors["q"], qwerty_anchors["p"])
    if (any((not isinstance(point, list) or len(point) != 2 or any((isinstance(value, bool) or not isinstance(value,
        (int, float)) for value in point)) for point in q_points))):
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
        box = (round(left * source.width / 1000.0), round(top * source.height / 1000.0),
            round(right * source.width / 1000.0), round(bottom * source.height / 1000.0))
        crop = source.convert("RGB").crop(box)
        width, height = crop.size
        if width < 40 or height < 40:
            continue
        pixels = crop.load()
        channel_values = [sorted((pixels[x, y][channel] for x in range(width) for y in range(height))) for channel
            in range(3)]
        midpoint = width * height // 2
        background = tuple(values[midpoint] for values in channel_values)

        def foreground(x: int, y: int) -> bool:
            pixel = pixels[x, y]
            return bool(max((abs(pixel[index] - background[index]) for index in range(3))) > 45
                or max(pixel) - min(pixel) > 55)

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
        near_columns = {x for length, x, start, end in candidates if abs(x - caret_x) <= 4
            and length >= 0.7 * run_length and (abs(start - caret_top) <= 5) and (abs(end - caret_bottom) <= 5)}
        if not 1 <= len(near_columns) <= 6:
            continue

        row_counts: list[int] = []
        for y in range(height):
            row_counts.append(sum(foreground(x, y) for x in range(width) if abs(x - caret_x) > 5))
        bands: list[tuple[int, int]] = []
        start: int | None = None
        previous: int | None = None
        for (y, count) in enumerate(row_counts):
            if count < 4:
                continue
            if start is None or previous is None or y - previous > 2:
                if start is not None and previous is not None:
                    bands.append((start, previous))
                start = y
            previous = y
        if start is not None and previous is not None:
            bands.append((start, previous))
        text_bands = [band for band in bands if band[0] > 3 and band[1] < height - 4 and band[1] - band[0] + 1 >= 4]
        if not text_bands:
            continue
        caret_center = (caret_top + caret_bottom) / 2.0
        line_index: int | None = None
        for (index, (band_top, band_bottom)) in enumerate(text_bands):
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


def _unique_clearable_ime_preedit(trusted_input: dict[str, Any], trusted_preedits: list[dict[str, Any]]) -> str:
    """Bind one adjacent IME composition to one typed input for clearing."""

    if len(trusted_preedits) != 1:
        return ""
    preedit_text = trusted_preedits[0].get("text")
    if not isinstance(preedit_text, str) or not preedit_text:
        return ""
    input_box = tuple(float(value) for value in trusted_input["input_bounds"])
    preedit_box = tuple(float(value) for value in trusted_preedits[0]["bounds"])
    overlaps_input = bool(_bounds_overlap_ratio(input_box, preedit_box) > 0 or _bounds_overlap_ratio(preedit_box,
        input_box) > 0)
    if overlaps_input:
        input_area = (input_box[2] - input_box[0]) * (input_box[3] - input_box[1])
        preedit_area = (preedit_box[2] - preedit_box[0]) * (preedit_box[3] - preedit_box[1])
        nested_empty_preedit = bool(trusted_input.get('text') == '' and _bounds_inside(preedit_box, input_box,
            tolerance=12) and (_bounds_overlap_ratio(preedit_box,
            input_box) >= 0.9) and (input_area > 0) and (preedit_area <= 0.6 * input_area))
        return preedit_text if nested_empty_preedit else ""
    return preedit_text if _horizontally_adjacent(input_box, preedit_box) else ""


def _unique_inline_ime_preedit_cue(trusted_input: dict[str, Any], trusted_preedits: list[dict[str, Any]], *,
    keyboard_input_mode: str) -> str:
    """Recover one unopposed pinyin preedit cue rendered inside the sole empty audited field."""

    cues = trusted_input.get("visible_editable_cues")
    if (keyboard_input_mode != 'chinese_pinyin' or trusted_input.get('text') != '' or trusted_preedits
        or (not isinstance(cues, list)) or (len(cues) != 1)):
        return ""
    cue = cues[0]
    if not isinstance(cue, str):
        return ""
    cue = cue.strip()
    non_text_cues = {'border', 'caret', 'cursor', 'focus border', 'focus ring', 'outline'}
    if (not cue or cue.casefold() in non_text_cues or cue == trusted_input.get('placeholder') or (cue
        in trusted_input.get('field_labels', ())) or (re.fullmatch("[a-z]+(?:'[a-z]+)*", cue, re.IGNORECASE) is None)):
        return ""
    return cue


def _next_field_key_element(key: dict[str, Any], *, source_field_id: str, target_field_id: str,
    target_field_label: str) -> dict[str, Any]:
    return _audited_element('next_field_key', 'input_next_field_key', key, states={'goal_relevant': True,
        'input_next_field_key': True, 'key_action': 'next', 'source_input_field_id': source_field_id,
        'target_input_field_id': target_field_id, 'target_input_field_label': target_field_label,
        'input_element_id': 'local_audited_input_1'}, evidence='唯一聚焦typed字段与完整Next键绑定下一依赖字段')


def _audited_element(suffix: str, meaning: str, source: Mapping[str, Any], *, states: Mapping[str, Any],
    evidence: str | Iterable[str], role: str='button', label: str | None=None, bounds_key: str='bounds') -> dict[str,
    Any]:
    state = {'fully_visible': True, **states, 'primary_input_geometry_verified': True,
        'geometry_audit_source': 'input_structure_audit'}
    return {'element_id': f'local_audited_{suffix}_1', 'role': role, 'meaning': meaning, 'label': source.get('label',
        '') if label is None else label, 'bounds': [part / 1000.0 for part in source[bounds_key]],
        'confidence': source['confidence'], 'states': state, 'evidence': [evidence] if isinstance(evidence,
        str) else list(evidence)[:6]}


def _trusted_ime_preedits(regions: list[Any], *, keyboard_visible: bool, keyboard_layout: str,
    keyboard_bounds: NormalizedBounds | None, qwerty_anchors: Mapping[str,
    Any] | None) -> list[dict[str, Any]]:
    """Keep only independently valid preedits and candidate targets."""

    result = []
    for region in regions:
        if (not isinstance(region, dict) or set(region) not in ({'region_id', 'bounds', 'text', 'confidence'},
            {'region_id', 'bounds', 'text', 'confidence', 'candidates'})
            or (not _valid_1000_bounds(region.get('bounds')))):
            continue
        try:
            confidence = _audit_confidence(region.get("confidence"), "IME预编辑区")
        except UISceneError:
            continue
        if confidence < 0.9:
            continue
        bounds = tuple(float(value) for value in region["bounds"])
        candidates = []
        raw_candidates = region.get("candidates", [])
        if isinstance(raw_candidates, list):
            for item in raw_candidates[:8]:
                if (not isinstance(item, dict) or set(item) != {'text', 'bounds', 'confidence',
                    'fully_visible'} or item.get('fully_visible') is not True
                    or (not _valid_1000_bounds(item.get('bounds')))):
                    continue
                text = str(item.get("text") or "").strip()
                try:
                    item_confidence = _audit_confidence(item.get("confidence"), "IME候选")
                except UISceneError:
                    continue
                candidate_bounds = tuple(float(value) for value in item["bounds"])
                if (text and len(text) <= 20 and (item_confidence >= 0.9)
                    and _ime_candidate_is_near_preedit(candidate_bounds, bounds, keyboard_visible=keyboard_visible,
                    keyboard_layout=keyboard_layout, keyboard_bounds=keyboard_bounds, qwerty_anchors=qwerty_anchors)):
                    candidates.append({'text': text, 'bounds': candidate_bounds, 'confidence': item_confidence})
        result.append({'text': str(region.get('text') or '').strip(), 'bounds': bounds, 'confidence': confidence,
            'candidates': candidates})
    return result


def _ime_candidate_is_near_preedit(candidate: NormalizedBounds, preedit: NormalizedBounds, *,
    keyboard_visible: bool, keyboard_layout: str, keyboard_bounds: NormalizedBounds | None,
    qwerty_anchors: Mapping[str, Any] | None) -> bool:
    gap = max(0.0, candidate[1] - preedit[3], preedit[1] - candidate[3])
    if _bounds_inside(candidate, preedit, tolerance=12) or gap <= 120:
        return True
    if (not (keyboard_visible and keyboard_layout == 'qwerty' and (keyboard_bounds is not None)
        and isinstance(qwerty_anchors, Mapping))):
        return False
    try:
        top_row_y = statistics.mean((float(qwerty_anchors["q"][1]), float(qwerty_anchors["p"][1])))
    except (KeyError, TypeError, ValueError):
        return False
    height = candidate[3] - candidate[1]
    return bool(15 <= height <= 100 and _bounds_inside(candidate, keyboard_bounds,
        tolerance=12) and (candidate[3] <= top_row_y - 8) and (candidate[1] <= keyboard_bounds[1] + min(180.0,
        0.35 * (keyboard_bounds[3] - keyboard_bounds[1]))))


def _audit_strings(value: Any, *, name: str, limit: int, allow_empty: bool) -> tuple[str, ...]:
    reject_if(
        not isinstance(value, list) or len(value) > limit or any((not isinstance(item,
        str) or (not allow_empty and (not item.strip() or len(item.strip()) > 120 or '\n' in item or ('\r'
        in item))) for item in value)),
        UISceneError(f"{name} 不符合输入结构协议。"),
    )
    return tuple(dict.fromkeys(item.strip()[:120] for item in value if item.strip()))


def _parse_audited_keyboard(value: Any, *, qwerty_row_snapper: Callable[[list[Image.Image] | tuple[Image.Image, ...],
    dict[str, Any]], dict[str, list[int]] | None] | None, qwerty_row_frames: list[Image.Image] | tuple[Image.Image,
    ...] | None) -> _AuditedKeyboard:
    reject_if(
        not isinstance(value, dict) or not _KEYBOARD_REQUIRED_FIELDS.issubset(value)
        or set(value) - _KEYBOARD_REQUIRED_FIELDS - _KEYBOARD_OPTIONAL_FIELDS,
        UISceneError("输入结构审计 keyboard 字段不符合协议。"),
    )
    keyboard = dict(value)
    visible = keyboard["visible"]
    layout = keyboard["layout"]
    input_mode = keyboard["input_mode"]
    case_mode = keyboard.get("case_mode", "unknown")
    reject_if(not isinstance(visible, bool), UISceneError("输入结构审计 keyboard.visible 必须是布尔值。"))
    reject_if(layout not in {'qwerty', 'numeric', 'symbol', 'unknown'}, UISceneError("输入结构审计 keyboard.layout 无效。"))
    reject_if(input_mode not in {'direct_latin', 'chinese_pinyin', 'unknown'}, UISceneError("输入结构审计 keyboard.input_mode 无效。"))
    reject_if(case_mode not in {'lower', 'upper', 'unknown'}, UISceneError("输入结构审计 keyboard.case_mode 无效。"))

    raw_anchors = keyboard.get("qwerty_anchors") if isinstance(keyboard.get("qwerty_anchors"), dict) else None
    snapped_anchors = None
    if (visible and layout == 'qwerty' and (raw_anchors is not None) and callable(qwerty_row_snapper)
        and (qwerty_row_frames is not None)):
        snapped_anchors = qwerty_row_snapper(qwerty_row_frames, raw_anchors)

    bounds = _keyboard_bounds_from_locally_snapped_qwerty_anchors(
        snapped_anchors) if snapped_anchors is not None else None
    if visible and bounds is None and _valid_1000_bounds(keyboard.get('bounds')):
        bounds = tuple(float(part) for part in keyboard["bounds"])
    if bounds is not None:
        width = bounds[2] - bounds[0]
        if bounds[3] - bounds[1] < 180 and width >= 300 and snapped_anchors:
            row_y = [float(point[1]) for point in snapped_anchors.values()]
            bounds = (bounds[0], min(bounds[1], min(row_y)), bounds[2], max(bounds[3], max(row_y)))
        if width < 300 or bounds[3] - bounds[1] < 180:
            bounds = None

    controls_revoked = visible and bounds is None
    if controls_revoked:
        layout = input_mode = case_mode = "unknown"
        keyboard.update(mode_switch=None, backspace_key=None, enter_key=None, case_switch=None, literal_keys=[],
            layout_switches=[])
        keyboard.pop("qwerty_anchors", None)
    elif (not visible and any((keyboard.get(key) not in (None, []) for key in ('bounds', 'mode_switch', 'backspace_key',
        'enter_key', 'case_switch', 'literal_keys', 'layout_switches')))):
        raise UISceneError("不可见键盘不能包含键位或切换控件。")

    application_bounds = bounds
    if _valid_1000_bounds(keyboard.get('bounds')):
        reported = tuple(float(part) for part in keyboard["bounds"])
        if reported[2] - reported[0] >= 300:
            application_bounds = reported
    return _AuditedKeyboard(keyboard, visible, layout, input_mode, case_mode, bounds, application_bounds, raw_anchors,
        snapped_anchors, controls_revoked)


def _label_count(labels: Iterable[str], expected: str) -> int:
    folded = expected.casefold()
    return sum(label.casefold() == folded for label in labels)


def _collect_audited_input_matches(application_inputs: list[Any], *, active_field_id: str, active_field_label: str,
    active_transaction_text: str, multiline_contract: bool, unique_typed_active_field: bool,
    application_keyboard_bounds: NormalizedBounds | None, trusted_preedits: list[dict[str, Any]],
    verified_input_lineage: TypedInputLineage | None, device_id: str, scene: UIScene, keyboard_input_mode: str,
    single_step_input_surface: dict[str, Any] | None) -> list[dict[str, Any]]:
    label_occurrences = sum((_label_count(item.get('field_labels', ()),
        active_field_label) for item in application_inputs if active_field_label and isinstance(item,
        dict) and isinstance(item.get('field_labels', ()), list)))
    tall_labeled_field = unique_typed_active_field and label_occurrences == 1
    matches: list[dict[str, Any]] = []
    for item in application_inputs:
        reject_if(
            not isinstance(item, dict) or not _INPUT_AUDIT_FIELDS.issubset(item)
            or set(item) - _INPUT_AUDIT_FIELDS - {'field_labels'} or (not isinstance(item.get('fully_visible'),
            bool)) or (not _valid_1000_bounds(item.get('bounds'))) or (not isinstance(item.get('text'), str)),
            UISceneError("应用输入结构字段不符合协议。"),
        )
        confidence = _audit_confidence(item["confidence"], "应用输入结构")
        cues = list(_audit_strings(item['visible_editable_cues'], name='visible_editable_cues', limit=4,
            allow_empty=True))
        labels = _audit_strings(item.get('field_labels', []), name='field_labels', limit=6, allow_empty=False)
        caret = item["caret_line_index"]
        reject_if(caret is not None and (isinstance(caret, bool) or not isinstance(caret, int) or (not 0 <= caret <= 30)), UISceneError("caret_line_index 必须是0..30整数或 null。"))
        if not item['fully_visible'] or confidence < 0.9:
            continue
        text = item["text"]
        placeholder = str(item.get("placeholder") or "").strip()
        bounds = tuple(float(part) for part in item["bounds"])
        surface_evidence = _same_frame_input_evidence(bounds, input_count=len(application_inputs), text=text,
            surface=single_step_input_surface)
        pending_shell = bool(verified_input_lineage is not None
            and verified_input_lineage.source == 'pending_verified_ime_candidate_action' and (active_field_id not
            in {'', 'unknown'}))
        if not any((text, placeholder, cues, pending_shell, surface_evidence)):
            continue
        max_height = 600 if multiline_contract or (tall_labeled_field and _label_count(labels,
            active_field_label) == 1) else 180
        width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        if (bounds[1] <= 10 or bounds[3] >= 990 or width < 240 or (not 20 <= height <= max_height)
            or (application_keyboard_bounds is not None and _bounds_overlap_ratio(bounds,
            application_keyboard_bounds) >= 0.25)):
            continue
        button = _optional_right_button(item["right_button"], input_bounds=bounds)
        input_bounds = [round(part) for part in bounds]
        if button is not None:
            input_bounds[2] = button["bounds"][0]
        if input_bounds[2] - input_bounds[0] < 120:
            continue
        match = {'text': text, 'placeholder': placeholder, 'visible_editable_cues': cues, 'caret_line_index': caret,
            'field_labels': labels, 'input_bounds': input_bounds, 'right_button': button,
            'same_frame_input_surface_evidence': surface_evidence, 'confidence': min(confidence,
            float(button['confidence']) if button else confidence)}
        match['pending_ime_candidate_state'] = _resolve_pending_ime_candidate_input_state(match, trusted_preedits,
            application_input_count=len(application_inputs), verified_input_lineage=verified_input_lineage,
            device_id=device_id, app_id=scene.app_id, screen_id=scene.screen_id, input_field_id=active_field_id,
            authorized_text=active_transaction_text, keyboard_input_mode=keyboard_input_mode)
        matches.append(match)
    return matches


def _unique_audited_input(matches: Iterable[dict[str, Any]], *, label: str='', text: str | None=None) -> dict[str,
    Any] | None:
    selected = [item for item in matches if (not label or _label_count(item['field_labels'],
        label) == 1) and (text is None or item['text'] == text)]
    return selected[0] if len(selected) == 1 else None


def _append_audited_input_element(elements: list[dict[str, Any]], audited_input: dict[str, Any], *, field_id: str,
    field_label: str, multiline: bool, active_clear_goal: bool, keyboard: _AuditedKeyboard,
    controls: _AuditedInputControls) -> None:
    step = controls.step
    auxiliary = any(getattr(controls, key) is not None for key in ("candidate", "literal", "enter", "layout", "case"))
    needs_auxiliary = bool(not active_clear_goal and step is not None and keyboard.visible
        and (step.kind == 'literal_key' or (controls.required_mode is not None
        and keyboard.input_mode != controls.required_mode) or (step.kind in {'direct_latin',
        'chinese_pinyin'} and keyboard.layout != 'qwerty') or (bool(step.required_case_mode)
        and keyboard.case_mode != step.required_case_mode)))
    states: dict[str, Any] = {'goal_relevant': bool(not controls.switch_is_goal and controls.next_field is None
        and (not auxiliary) and (not needs_auxiliary) and (not keyboard.controls_revoked)), 'fully_visible': True,
        'value': audited_input['text']}
    if field_id:
        states["input_field_id"] = field_id
        if controls.next_field is None:
            states["input_multiline"] = multiline
    if field_label:
        states["input_field_label"] = field_label
    for key in ('verified_trailing_newline', 'local_caret_line_index', 'clear_extra_delete_units'):
        value = audited_input.get(key)
        if value is True or (isinstance(value, int) and (not isinstance(value, bool)) and (value > 0)):
            states[key] = value
    if not keyboard.visible:
        states["soft_keyboard_visible"] = False
    if audited_input['placeholder']:
        states["placeholder"] = audited_input["placeholder"]
    if keyboard.visible:
        states.update(focused=True, keyboard_layout=keyboard.layout, keyboard_input_mode=keyboard.input_mode,
            keyboard_case_mode=keyboard.case_mode)
    if controls.qwerty is not None:
        states["keyboard_geometry"] = controls.qwerty
    elif controls.backspace is not None:
        states['keyboard_geometry'] = {'type': 'generic', 'anchors': {'backspace': controls.backspace['center']},
            'source': 'input_structure_audit'}
    if controls.candidate is not None and step is not None:
        states.update(ime_preedit_text=controls.preedit, ime_exact_candidate_text=step.segment)
    elif controls.clearable_preedit:
        states["ime_preedit_text"] = controls.clearable_preedit

    evidence = list(dict.fromkeys((*audited_input['field_labels'], *audited_input['visible_editable_cues'],
        *audited_input.get('same_frame_input_surface_evidence', ()))))
    text = audited_input["text"]
    if text:
        evidence.insert(0, f"应用输入框当前文字：{text}")
        templates = {'lineage_visual_text': '视觉折行转写：{}；本地逐键连续性逐字核对通过',
            'lineage_visible_cue_text': '输入状态切换后同一应用输入区域仍逐字可见：{}；本地同值连续性核对通过',
            'lineage_persisted_visible_cue_text': '跨会话同一应用输入区域仍逐字可见：{}；持久回执连续性核对通过',
            'lineage_pending_text_prefix': 'pending typed连续性、授权payload与唯一预编辑后缀共同确认已提交前缀：{}',
            'lineage_ime_candidate_committed_value': '候选点击回执、typed字段、授权payload与动作后同字段精确片段共同确认已提交中文：{}；残留预测栏未作为预编辑',
            'authorized_prefix_visible_cue_text': 'typed字段内唯一可见文字逐字匹配授权payload前缀：{}；同帧无IME预编辑',
            'clear_goal_visible_cue_text': '清空目标的同帧唯一已提交可见文字：{}；额外视觉行仅作为保守退格单位'}
        evidence.extend((template.format(audited_input[key]) for key,
            template in templates.items() if isinstance(audited_input.get(key), str)))
        if audited_input.get('verified_trailing_newline') is True:
            evidence.append("已验证换行动作、同一typed输入框、精确可见前缀与下一行光标一致")
        if isinstance(audited_input.get('local_caret_line_index'), int):
            evidence.append(f'模型确认可见光标后，本地校准像素定位光标视觉行：{audited_input['local_caret_line_index']}')
    elif audited_input['placeholder']:
        evidence.insert(0, f"应用输入框为空，占位提示：{audited_input['placeholder']}")
    elif audited_input.get('same_frame_input_surface_evidence'):
        evidence.insert(0, "输入结构审计确认当前输入框为空；同帧场景只证明唯一可见输入表面与其重合")
    if controls.clearable_preedit:
        evidence.append("唯一相邻输入法预编辑串已绑定当前typed输入框，可清理文字：" f"{controls.clearable_preedit}")
    if not keyboard.visible:
        evidence.append(AUDITED_SOFT_KEYBOARD_HIDDEN_EVIDENCE)
    elements.append(_audited_element('input', 'application_text_input', audited_input, role='input',
        label=text or audited_input['placeholder'], bounds_key='input_bounds', states=states, evidence=evidence))
    if audited_input['right_button'] is not None:
        elements.append(_audited_element('adjacent_button', 'adjacent_input_utility', audited_input['right_button'],
            states={'goal_relevant': False}, evidence='应用输入结构的相邻独立控件；不具备目标权限'))


def _append_audited_input_controls(elements: list[dict[str, Any]], *, controls: _AuditedInputControls,
    active_field_id: str, predecessor_field_id: str, active_field_label: str) -> None:
    step = controls.step
    next_field = controls.next_field
    if next_field is not None:
        elements.append(_next_field_key_element(next_field, source_field_id=predecessor_field_id,
            target_field_id=active_field_id, target_field_label=active_field_label))
    if step is not None:
        common = {'prior_input_value': step.current_text, 'input_element_id': 'local_audited_input_1'}
        specs = (('candidate', 'ime_candidate', 'ime_exact_candidate', {'ime_candidate': True,
            'expected_input_value': step.expected_value, 'pinyin': controls.preedit},
            '输入结构审计确认当前输入法组合的唯一逐字候选：' + step.segment), ('literal', 'literal_key', 'input_exact_literal_key',
            {'input_literal_key': True, 'key_value': step.segment, 'expected_input_value': step.expected_value},
            '输入结构审计确认下一字符对应唯一完整可见键位'), ('enter', 'enter_key', 'input_exact_enter_key', {'input_enter_key': True,
            'key_action': 'newline', 'key_value': '\n', 'expected_input_value': step.expected_value,
            'input_field_id': active_field_id}, '输入结构审计确认当前多行字段的唯一完整可见换行键'), ('layout', 'keyboard_layout_switch',
            'switch_keyboard_layout', {'keyboard_layout_switch': True,
            'current_layout': controls.layout['current_layout'] if controls.layout else '',
            'target_layout': controls.layout['target_layout'] if controls.layout else '',
            'next_input_value': step.segment}, '输入结构审计确认方向明确的键盘布局切换键'), ('case', 'keyboard_case_switch',
            'switch_keyboard_case', {'keyboard_case_switch': True,
            'current_mode': controls.case['current_mode'] if controls.case else '',
            'target_mode': controls.case['target_mode'] if controls.case else ''}, '输入结构审计确认方向明确的大小写切换键'))
        for (key, suffix, meaning, extra_states, evidence) in specs:
            item = getattr(controls, key)
            if item is not None:
                elements.append(_audited_element(suffix, meaning, item,
                    label=item['text'] if key == 'candidate' else None, states={'goal_relevant': True, **common,
                    **extra_states}, evidence=evidence))
    mode = controls.mode
    if mode is not None:
        elements.append(_audited_element('keyboard_mode_switch', 'switch_keyboard_input_mode', mode,
            states={'goal_relevant': controls.switch_is_goal, 'keyboard_input_mode_switch': True,
            'current_mode': mode['current_mode'], 'target_mode': mode['target_mode'],
            'prior_input_value': step.current_text if step else '', 'next_input_value': step.segment if step else '',
            'input_element_id': 'local_audited_input_1'}, evidence='键盘区域内方向明确的独立输入模式切换键'))


def _same_frame_input_evidence(input_bounds: NormalizedBounds, *, input_count: int, text: str,
    surface: dict[str, Any] | None) -> tuple[str, ...]:
    if input_count != 1 or text or (not isinstance(surface, dict)):
        return ()
    surface_bounds = surface.get("bounds")
    evidence = surface.get("evidence")
    if not _valid_1000_bounds(surface_bounds) or not isinstance(evidence, (list, tuple)):
        return ()
    other = tuple(float(value) for value in surface_bounds)
    if _bounds_overlap_ratio(input_bounds, other) < 0.85 or _bounds_overlap_ratio(other, input_bounds) < 0.85:
        return ()
    return tuple(str(value).strip()[:200] for value in evidence if str(value).strip())


def _optional_right_button(value: Any, *, input_bounds: NormalizedBounds) -> dict[str, Any] | None:
    if (not isinstance(value, dict) or set(value) != {'label', 'bounds',
        'confidence'} or (not _valid_1000_bounds(value.get('bounds')))):
        return None
    try:
        confidence = _audit_confidence(value.get("confidence"), "right_button")
    except UISceneError:
        return None
    label = str(value.get("label") or "").strip()
    bounds = tuple(float(part) for part in value["bounds"])
    width = input_bounds[2] - input_bounds[0]
    if (confidence < 0.9 or not label or (not _bounds_inside(bounds, input_bounds,
        tolerance=20)) or (bounds[0] <= input_bounds[0] + 0.55 * width) or (_vertical_overlap_ratio(bounds,
        input_bounds) < 0.8)):
        return None
    return {'label': label, 'bounds': [round(part) for part in bounds], 'confidence': confidence}


def _reconcile_verified_input_lineage(trusted_input: dict[str, Any], trusted_preedits: list[dict[str, Any]],
    lineage: TypedInputLineage, *, device_id: str, scene: UIScene, input_field_id: str, active_text: str,
    multiline: bool, frame: Image.Image | None, keyboard_input_mode: str) -> dict[str, Any]:
    raw_text = trusted_input["text"]
    bounds = tuple(float(part) / 1000.0 for part in trusted_input["input_bounds"])
    cues = tuple(trusted_input["visible_editable_cues"])
    if (keyboard_input_mode == 'direct_latin' and lineage.exact_value not in cues
        and _adjacent_exact_preedit_cue(trusted_input, trusted_preedits, lineage.exact_value)):
        cues = (*cues, lineage.exact_value)
    identity = {'device_id': device_id, 'app_id': scene.app_id, 'screen_id': scene.screen_id, 'raw_value': raw_text,
        'input_bounds': bounds}
    checks = (('verified_trailing_newline', lambda: lineage_matches_trailing_newline_cue(lineage, **identity,
        visible_editable_cues=cues, caret_line_index=trusted_input.get('caret_line_index'),
        input_field_id=input_field_id, current_frame=frame)), ('lineage_visible_cue_text',
        lambda: lineage.matches_pending_input_state_cue(**identity, visible_editable_cues=cues,
        input_field_id=input_field_id)), ('lineage_persisted_visible_cue_text',
        lambda: lineage_matches_persisted_surface_cue(lineage, **identity, visible_editable_cues=cues,
        current_frame=frame)), ('lineage_visual_text', lambda: not multiline and lineage_matches_visual(lineage,
        **identity, current_frame=frame)))
    result = trusted_input
    for (key, predicate) in checks:
        if predicate():
            result = dict(result)
            result[key] = (
                True
                if key == "verified_trailing_newline"
                else (raw_text if key == "lineage_visual_text" else lineage.exact_value)
            )
            result["text"] = lineage.exact_value
            break
    if result['text'] == '':
        prefix = lineage.pending_text_committed_prefix(device_id=device_id, app_id=scene.app_id,
            screen_id=scene.screen_id, authorized_text=active_text, raw_value='',
            preedit_text=_unique_clearable_ime_preedit(result, trusted_preedits), input_bounds=bounds,
            input_field_id=input_field_id)
        if prefix is not None:
            result = dict(result)
            result["lineage_pending_text_prefix"] = prefix
            result["text"] = prefix
    return result


def _plan_audited_input_controls(*, trusted_input: dict[str, Any] | None, predecessor_input: dict[str, Any] | None,
    trusted_preedits: list[dict[str, Any]], goal: generic_goal_domain.ActiveVisualGoal, active_clear_goal: bool,
    active_field_id: str, active_multiline: bool, keyboard: dict[str, Any], keyboard_visible: bool,
    keyboard_layout: str, keyboard_input_mode: str, keyboard_case_mode: str, keyboard_bounds: tuple[float, float, float,
    float] | None, snapped_anchors: dict[str, list[int]] | None, switch_is_goal: bool) -> _AuditedInputControls:
    """Derive the sole next typed-input control set from audited facts."""

    step = None
    if trusted_input is not None and goal.has_explicit_text and (not active_clear_goal):
        target = goal.transaction_text or goal.explicit_text
        try:
            step = plan_next_verified_input(target, trusted_input["text"])
        except (ValueError, VerifiedTextTransactionError):
            pass

    exact_ime_candidate = None
    exact_ime_preedit = ""
    if step is not None and step.kind in {'chinese_pinyin', 'direct_latin'}:
        matches = [item for item in trusted_preedits if (re.sub('[^a-z]', '',
            item['text'].casefold()) == step.pinyin if step.kind == 'chinese_pinyin' else item['text'] == step.segment)]
        candidates = [candidate for item in matches for candidate
            in item['candidates'] if candidate['text'] == step.segment]
        if len(matches) == len(candidates) == 1:
            exact_ime_candidate = candidates[0]
            exact_ime_preedit = step.pinyin if step.kind == "chinese_pinyin" else matches[0]["text"]
        elif len(matches) == 1:
            raise UISceneError("有用输入法预编辑必须提供唯一逐字候选几何，不能转为清除。")

    qwerty_geometry = None
    raw_anchors = keyboard.get("qwerty_anchors")
    if raw_anchors is not None:
        reject_if(not keyboard_visible or keyboard_layout != 'qwerty' or keyboard_bounds is None, UISceneError("QWERTY anchors 必须绑定完整可见的 QWERTY 键盘。"))
        qwerty_geometry = _validated_qwerty_keyboard_geometry(snapped_anchors or raw_anchors,
            keyboard_bounds=keyboard_bounds, locally_snapped=snapped_anchors is not None)
    keyboard_controls = _validated_keyboard_controls(keyboard, keyboard_bounds=keyboard_bounds,
        keyboard_layout=keyboard_layout, keyboard_input_mode=keyboard_input_mode,
        keyboard_case_mode=keyboard_case_mode, snapped_anchors=snapped_anchors)
    backspace = keyboard_controls['backspace']
    clearable_preedit = ""
    if (trusted_input is not None and active_field_id and (goal.transaction_text or active_clear_goal)
        and keyboard_visible and (keyboard_bounds is not None) and (qwerty_geometry is not None or backspace
        is not None)):
        clearable_preedit = _unique_clearable_ime_preedit(trusted_input,
            trusted_preedits) or _unique_inline_ime_preedit_cue(trusted_input, trusted_preedits,
            keyboard_input_mode=keyboard_input_mode)

    enter_key = keyboard_controls['enter']
    next_field_key = enter_key if predecessor_input is not None and enter_key is not None and (enter_key[
        'key_action'] == 'next') else None
    required_mode = required_keyboard_input_mode_for_step(step) if step is not None else None
    needs_mode_switch = bool(not active_clear_goal and (not clearable_preedit) and (exact_ime_candidate is None)
        and (step is not None) and (required_mode is not None) and keyboard_visible and (keyboard_layout == 'qwerty')
        and (keyboard_input_mode in {'direct_latin', 'chinese_pinyin'}) and (keyboard_input_mode != required_mode))
    switch_is_goal = switch_is_goal or needs_mode_switch
    mode_switch = keyboard_controls['mode']
    reject_if(needs_mode_switch and (mode_switch is None or mode_switch['target_mode'] != required_mode), UISceneError("模式切换键未绑定下一确定性文字分段所需方向。"))

    literal_keys = keyboard_controls['literal_keys']
    targets = set(_input_audit_literal_key_targets(goal.observation_context,
        current_input_text=trusted_input['text'] if trusted_input is not None else None))
    literal_keys = [item for item in literal_keys if item["value"] in targets]
    if keyboard_layout == 'qwerty' and qwerty_geometry is not None:
        literal_keys = [item for item in literal_keys
            if not _literal_key_conflicts_with_qwerty(item, qwerty_geometry=qwerty_geometry)]
    layout_switches = keyboard_controls['layout_switches']
    case_switch = keyboard_controls['case']
    exact_literal = exact_enter = exact_layout = exact_case = None
    if step is not None:
        if required_mode is not None and keyboard_input_mode != required_mode:
            if keyboard_layout != 'qwerty':
                exact_layout = _select_keyboard_layout_switch_for_target(layout_switches,
                    current_layout=keyboard_layout, target_layout='qwerty')
        elif step.kind in {'direct_latin', 'chinese_pinyin'} and keyboard_layout != 'qwerty':
            exact_layout = _select_keyboard_layout_switch_for_target(layout_switches, current_layout=keyboard_layout,
                target_layout='qwerty')
        elif step.kind == 'literal_key' and step.segment == '\n':
            if active_multiline and enter_key is not None and (enter_key['key_action'] == 'newline'):
                exact_enter = enter_key
        elif step.kind == 'literal_key':
            exact = [item for item in literal_keys if item["value"] == step.segment]
            if len(exact) == 1:
                exact_literal = exact[0]
            else:
                desired_layout = _preferred_keyboard_layout(step.segment)
                if desired_layout != keyboard_layout:
                    exact_layout = _select_keyboard_layout_switch_for_target(layout_switches,
                        current_layout=keyboard_layout, target_layout=desired_layout)
        elif (step.kind == 'direct_latin' and step.required_case_mode
            and (keyboard_case_mode != step.required_case_mode) and (case_switch is not None)
            and (case_switch['target_mode'] == step.required_case_mode)):
            exact_case = case_switch
    reject_if(
        trusted_input is not None and keyboard_visible and (keyboard_layout == 'qwerty') and (keyboard_input_mode
        in {'direct_latin', 'chinese_pinyin'}) and goal.has_explicit_text and (not switch_is_goal) and (step
        is not None) and (step.kind in {'direct_latin', 'chinese_pinyin'}) and (exact_case is None)
        and (qwerty_geometry is None) and (not active_clear_goal),
        UISceneError("文字输入授权要求本轮输入结构审计提供有效 QWERTY anchors。"),
    )
    return _AuditedInputControls(step=step, candidate=exact_ime_candidate, preedit=exact_ime_preedit,
        qwerty=qwerty_geometry, backspace=backspace, clearable_preedit=clearable_preedit, next_field=next_field_key,
        required_mode=required_mode, switch_is_goal=switch_is_goal, mode=mode_switch, literal=exact_literal,
        enter=exact_enter, layout=exact_layout, case=exact_case)


def _apply_input_structure_audit(scene: UIScene, raw: str, *, fingerprint: str, goal_context: dict[str, Any],
    verified_input_lineage: TypedInputLineage | None=None, device_id: str | None=None,
    lineage_frame: Image.Image | None=None, qwerty_row_snapper: Callable[[list[Image.Image] | tuple[Image.Image, ...],
    dict[str, Any]], dict[str, list[int]] | None] | None=None, qwerty_row_frames: list[Image.Image] | tuple[Image.Image,
    ...] | None=None, single_step_input_surface: dict[str, Any] | None=None) -> UIScene:
    try:
        goal = _goal_view(goal_context)
        payload = _extract_json_object(raw)
        reject_if(set(payload) != {'protocol_version', 'application_inputs', 'ime_preedit_regions', 'keyboard'}, UISceneError("输入结构审计包含协议外字段。"))
        reject_if(payload.get('protocol_version') != INPUT_STRUCTURE_AUDIT_VERSION, UISceneError("输入结构审计协议版本不匹配。"))
        application_inputs = payload.get("application_inputs")
        ime_preedit_regions = payload.get("ime_preedit_regions")
        keyboard = payload.get("keyboard")
        reject_if(not isinstance(application_inputs, list) or len(application_inputs) > 4, UISceneError("输入结构审计 application_inputs 必须是最多4项的数组。"))
        reject_if(not isinstance(ime_preedit_regions, list) or len(ime_preedit_regions) > 4, UISceneError("输入结构审计 ime_preedit_regions 必须是最多4项的数组。"))
        audited_keyboard = _parse_audited_keyboard(keyboard, qwerty_row_snapper=qwerty_row_snapper,
            qwerty_row_frames=qwerty_row_frames)
        keyboard = audited_keyboard.payload
        keyboard_visible = audited_keyboard.visible
        keyboard_layout = audited_keyboard.layout
        keyboard_input_mode = audited_keyboard.input_mode
        keyboard_case_mode = audited_keyboard.case_mode
        keyboard_bounds = audited_keyboard.bounds
        application_keyboard_bounds = audited_keyboard.application_bounds
        raw_audited_qwerty_anchors = audited_keyboard.raw_qwerty_anchors
        locally_snapped_qwerty_anchors = audited_keyboard.snapped_qwerty_anchors
        keyboard_controls_revoked = audited_keyboard.controls_revoked
        if keyboard_controls_revoked:
            ime_preedit_regions = []

        trusted_preedits = _trusted_ime_preedits(ime_preedit_regions, keyboard_visible=keyboard_visible,
            keyboard_layout=keyboard_layout, keyboard_bounds=application_keyboard_bounds,
            qwerty_anchors=locally_snapped_qwerty_anchors or raw_audited_qwerty_anchors)
        active_field_id, active_field_label, active_multiline = goal.field
        active_transaction_text = goal.transaction_text
        explicit_input_text = goal.explicit_text
        multiline_input_contract = bool(active_multiline or "\n" in explicit_input_text or "\r" in explicit_input_text)
        unique_typed_active_field = goal.unique_typed_field
        matches = _collect_audited_input_matches(application_inputs, active_field_id=active_field_id,
            active_field_label=active_field_label, active_transaction_text=active_transaction_text,
            multiline_contract=multiline_input_contract, unique_typed_active_field=unique_typed_active_field,
            application_keyboard_bounds=application_keyboard_bounds, trusted_preedits=trusted_preedits,
            verified_input_lineage=verified_input_lineage, device_id=str(device_id or ''), scene=scene,
            keyboard_input_mode=keyboard_input_mode, single_step_input_surface=single_step_input_surface)

        switch_is_goal = goal.mode_switch_requested
        active_clear_goal = goal.clear_requested
        trusted_input = _unique_audited_input(matches, label=active_field_label)
        if (trusted_input is not None and trusted_input.get('pending_ime_candidate_state') is not None
            and (verified_input_lineage is not None)):
            resolved_ime_state = trusted_input["pending_ime_candidate_state"]
            committed_preedit = resolved_ime_state["consumed_preedit"]
            trusted_input = dict(trusted_input)
            trusted_input["lineage_ime_candidate_committed_value"] = resolved_ime_state["committed_value"]
            trusted_input["text"] = resolved_ime_state["committed_value"]
            trusted_preedits = [item for item in trusted_preedits if item is not committed_preedit]
        predecessor_field_id, predecessor_field_label, predecessor_text = goal.predecessor
        predecessor_input = _unique_audited_input(matches, label=predecessor_field_label,
            text=predecessor_text) if predecessor_field_id else None
        if trusted_input is not None and trusted_input.get('caret_line_index') is None:
            local_caret_line_index = _locally_detect_caret_visual_row(
                trusted_input,
                frames=(tuple(qwerty_row_frames) if qwerty_row_frames is not None else None),
                # Raw anchors bound pixel search only; action geometry still requires multi-frame row snapping.
                qwerty_anchors=(locally_snapped_qwerty_anchors or raw_audited_qwerty_anchors),
                allow_goal_bound_local_detection=bool(
                    active_clear_goal or (isinstance(active_transaction_text, str) and "\n" in active_transaction_text)
                ),
            )
            if local_caret_line_index is not None:
                trusted_input = dict(trusted_input)
                trusted_input["caret_line_index"] = local_caret_line_index
                trusted_input["local_caret_line_index"] = local_caret_line_index
        if trusted_input is not None and active_clear_goal and (not trusted_input['text']):
            clear_cue, extra_clear_units = _clear_goal_unique_committed_cue(trusted_input, trusted_preedits,
                keyboard_input_mode=keyboard_input_mode)
            if clear_cue:
                trusted_input = dict(trusted_input)
                trusted_input["clear_goal_visible_cue_text"] = clear_cue
                trusted_input["clear_extra_delete_units"] = extra_clear_units
                trusted_input["text"] = clear_cue
        if (trusted_input is not None and active_clear_goal and isinstance(trusted_input.get('text'),
            str) and trusted_input['text'] and isinstance(trusted_input.get('caret_line_index'), int)):
            extra_clear_units = max(0, trusted_input['caret_line_index'] - trusted_input['text'].count('\n'))
            if extra_clear_units > 0:
                trusted_input = dict(trusted_input)
                trusted_input["clear_extra_delete_units"] = extra_clear_units
        if trusted_input is not None and (not trusted_input['text']) and (verified_input_lineage is None):
            authorized_prefix_cue = _authorized_exact_committed_prefix_cue(trusted_input, trusted_preedits, goal=goal,
                keyboard_input_mode=keyboard_input_mode)
            if authorized_prefix_cue:
                trusted_input = dict(trusted_input)
                trusted_input["authorized_prefix_visible_cue_text"] = authorized_prefix_cue
                trusted_input["text"] = authorized_prefix_cue
        if trusted_input is not None and verified_input_lineage is not None:
            trusted_input = _reconcile_verified_input_lineage(trusted_input, trusted_preedits, verified_input_lineage,
                device_id=str(device_id or ''), scene=scene, input_field_id=active_field_id,
                active_text=active_transaction_text, multiline=active_multiline, frame=lineage_frame,
                keyboard_input_mode=keyboard_input_mode)
        controls = _plan_audited_input_controls(trusted_input=trusted_input, predecessor_input=predecessor_input,
            trusted_preedits=trusted_preedits, goal=goal, active_clear_goal=active_clear_goal,
            active_field_id=active_field_id, active_multiline=active_multiline, keyboard=keyboard,
            keyboard_visible=keyboard_visible, keyboard_layout=keyboard_layout, keyboard_input_mode=keyboard_input_mode,
            keyboard_case_mode=keyboard_case_mode, keyboard_bounds=keyboard_bounds,
            snapped_anchors=locally_snapped_qwerty_anchors, switch_is_goal=switch_is_goal)
        rendered_input = trusted_input or (predecessor_input if controls.next_field is not None else None)

        focus_only_input = None
        if (trusted_input is None and controls.next_field is None and (controls.mode is None
            or not controls.switch_is_goal) and (goal.transaction_text or goal.target_only)):
            focus_only_input = _focus_only_compact_input_surface(single_step_input_surface,
                keyboard_visible=keyboard_visible)

        value = scene.to_dict()
        elements: list[dict[str, Any]] = []
        for element in value.get('elements') or []:
            if not isinstance(element, dict):
                continue
            element = dict(element)
            element["states"] = dict(element.get("states") or {})
            element["states"]["goal_relevant"] = False
            # Compact input proposals never survive the typed audit, including its early-return path.
            if element.get('role') == 'input' or element.get('meaning') == 'application_text_input':
                continue
            elements.append(element)
        if (trusted_input is None and controls.next_field is None and (controls.mode is None
            or not controls.switch_is_goal)):
            if focus_only_input is not None:
                elements.append(focus_only_input)
            value["elements"] = elements
            value["summary"] = (
                "专用typed输入状态账本尚未建立；" "仅保留唯一粗编辑面用于聚焦，不授权文字、清空或发送。"
                if focus_only_input is not None
                else "typed输入状态账本未建立；compact输入摘要与输入转写不参与判断。"
            )
            return UIScene.from_dict(value, coordinate_scale=1.0, stable_override=True,
                fingerprint_override=fingerprint)
        if rendered_input is not None:
            _append_audited_input_element(elements, rendered_input,
                field_id=predecessor_field_id if controls.next_field is not None else active_field_id,
                field_label=predecessor_field_label if controls.next_field is not None else active_field_label,
                multiline=active_multiline, active_clear_goal=active_clear_goal, keyboard=audited_keyboard,
                controls=controls)
        _append_audited_input_controls(elements, controls=controls, active_field_id=active_field_id,
            predecessor_field_id=predecessor_field_id, active_field_label=active_field_label)
        value["elements"] = elements
        value["summary"] = "输入状态仅见typed输入账本；compact输入摘要与输入转写不参与判断。"
        return UIScene.from_dict(value, coordinate_scale=1.0, stable_override=True, fingerprint_override=fingerprint)
    except (UISceneError, ValueError, TypeError) as exc:
        raise VisionAgentError(f"输入结构只读审计结果不符合协议：{exc}") from exc


def _audit_confidence(value: Any, field_name: str) -> float:
    reject_if(isinstance(value, bool) or not isinstance(value, (int, float)), UISceneError(f"{field_name} confidence 格式无效。"))
    confidence = float(value)
    reject_if(not 0.0 <= confidence <= 1.0, UISceneError(f"{field_name} confidence 超出0..1。"))
    return confidence


_QWERTY_ANCHOR_KEYS = frozenset({'q', 'p', 'a', 'l', 'z', 'm', 'backspace'})


def _qwerty_anchor_points(value: Any) -> dict[str, list[float]] | None:
    if not isinstance(value, dict) or set(value) != _QWERTY_ANCHOR_KEYS:
        return None
    if any(not isinstance(point, (list, tuple)) or len(point) != 2 or any(isinstance(part,
        bool) or not isinstance(part, (int, float)) for part in point) for point in value.values()):
        return None
    return {key: [float(value[key][0]), float(value[key][1])] for key in _QWERTY_ANCHOR_KEYS}


def _keyboard_bounds_from_locally_snapped_qwerty_anchors(value: Any) -> NormalizedBounds | None:
    """Rebuild a keyboard envelope from independently stabilized QWERTY rows."""

    anchors = _qwerty_anchor_points(value)
    if anchors is None or any(not 0 <= coordinate <= 1000 for point in anchors.values() for coordinate in point):
        return None
    try:
        normalized = {key: [round(point[0]), round(point[1])] for key, point in anchors.items()}
        qwerty_keyboard_config_from_anchors(normalized)
        horizontal_pitch = min((anchors[right][0] - anchors[left][0]) / gaps for left, right, gaps
            in (('q', 'p', 9.0), ('a', 'l', 8.0), ('z', 'm', 6.0)))
        row_y = ((anchors['q'][1] + anchors['p'][1]) / 2.0,
            (anchors['a'][1] + anchors['l'][1]) / 2.0,
            (anchors['z'][1] + anchors['m'][1] + anchors['backspace'][1]) / 3.0)
        vertical_pitch = min(row_y[1] - row_y[0], row_y[2] - row_y[1])
        if horizontal_pitch <= 0 or vertical_pitch <= 0:
            return None
        bounds = (
            max(0.0, min(anchors[key][0] for key in ('q', 'a', 'z')) - 0.75 * horizontal_pitch),
            max(0.0, row_y[0] - 0.75 * vertical_pitch),
            min(1000.0, max(anchors[key][0] for key in ('p', 'l', 'm', 'backspace')) + 0.75 * horizontal_pitch),
            min(1000.0, row_y[2] + 2.0 * vertical_pitch),
        )
        if bounds[2] - bounds[0] < 300 or bounds[3] - bounds[1] < 180:
            return None
        return bounds
    except (KeyError, TypeError, ValueError, WorkflowNotReady):
        return None


def _validated_qwerty_keyboard_geometry(value: Any, *, keyboard_bounds: NormalizedBounds,
    locally_snapped: bool=False) -> dict[str, Any]:
    """Mint a locally checked execution profile from current-frame facts."""

    points = _qwerty_anchor_points(value)
    reject_if(points is None, UISceneError("QWERTY anchors 必须精确包含七个数值点。"))
    normalized = {key: [round(x), round(y)] for key, (x, y) in sorted(points.items())}
    for key, (x, y) in points.items():
        inside_x = keyboard_bounds[0] - 20 <= x <= keyboard_bounds[2] + 20
        inside_y = keyboard_bounds[1] - 20 <= y <= keyboard_bounds[3] + 20
        reject_if(not inside_x or (not locally_snapped and not inside_y),
            UISceneError(f"QWERTY anchor {key} 不在已审计键盘区域内。"))
    try:
        qwerty_keyboard_config_from_anchors(normalized)
    except WorkflowNotReady as exc:
        raise UISceneError(f"QWERTY anchors 未通过本地布局校验：{exc}") from exc
    return {'type': 'qwerty', 'anchors': normalized, 'source': 'input_structure_audit'}


def _locally_snapped_keyboard_enter_key(value: Any, *, anchors: dict[str, list[int]], keyboard_bounds: tuple[float,
    float, float, float] | None) -> dict[str, Any] | None:
    fields = {"label", "bounds", "confidence", "fully_visible", "key_action"}
    if (keyboard_bounds is None or not isinstance(value,
        dict) or set(value) != fields or (value.get('fully_visible') is not True) or (value.get('key_action') not
        in {'newline', 'next'})):
        return None
    action, label = value['key_action'], str(value.get('label') or '').strip()
    normalized = re.sub(r"\s+", "", label).casefold()
    newline_label = any(glyph in label for glyph in ('↵', '⏎', '⤶', '⮐')) or normalized in {
        'enter', 'return', '回车', '换行'}
    next_label = any(glyph in label for glyph in ('→', '↦', '➡', '⏭')) or normalized in {
        'next', '下一步', '下一个', '下一项'}
    if not (action == 'newline' and (newline_label or not normalized) or (action == 'next' and next_label)):
        return None
    confidence = _audit_confidence(value.get("confidence"), "enter_key")
    raw_bounds = value.get("bounds")
    if (confidence < 0.9 or not isinstance(raw_bounds, (list, tuple)) or len(raw_bounds) != 4 or any((isinstance(part,
        bool) or not isinstance(part, (int, float)) for part in raw_bounds))):
        return None
    left, top, right, bottom = (float(part) for part in raw_bounds)
    if not (0 <= left < right <= 1000 and math.isfinite(top) and math.isfinite(bottom) and (top < bottom)):
        return None
    try:
        profile = qwerty_keyboard_config_from_anchors(anchors)
        rows = profile['rows']
        horizontal_pitch = float(rows[0]['x_step']) * 1000
        row_y = [float(row["y"]) * 1000 for row in rows]
        vertical_pitch = statistics.mean((second - first for first, second in zip(row_y, row_y[1:])))
        backspace_x = float(profile['backspace_x_ratio']) * 1000
    except (KeyError, TypeError, ValueError, WorkflowNotReady):
        return None
    if (not 45 <= horizontal_pitch <= 130 or not 35 <= vertical_pitch <= 140
        or abs((left + right) / 2 - backspace_x) > max(2 * horizontal_pitch, 180)):
        return None
    center_y = row_y[2] + vertical_pitch
    snapped = (max(keyboard_bounds[0], backspace_x - 0.65 * horizontal_pitch), max(keyboard_bounds[1],
        center_y - 0.45 * vertical_pitch), min(keyboard_bounds[2], backspace_x + 0.65 * horizontal_pitch),
        min(keyboard_bounds[3], center_y + 0.45 * vertical_pitch))
    if not _valid_1000_bounds(snapped) or not _bounds_inside(snapped, keyboard_bounds, tolerance=0):
        return None
    return {'label': label or '↵', 'bounds': [round(part) for part in snapped], 'confidence': confidence,
        'key_action': action}


def _preferred_keyboard_layout(character: str) -> str:
    try:
        return preferred_keyboard_layout(character)
    except VerifiedTextTransactionError as exc:
        raise UISceneError(str(exc)) from exc


def _select_keyboard_layout_switch_for_target(layout_switches: list[dict[str, Any]], *, current_layout: str,
    target_layout: str) -> dict[str, Any] | None:
    """Select the unique visible edge that moves toward the requested layout."""

    next_layout = next_keyboard_layout_towards(current_layout, target_layout)
    if next_layout is None:
        return None
    direct = [item for item in layout_switches if item.get('current_layout') == current_layout
        and item.get('target_layout') == target_layout]
    if len(direct) == 1:
        return direct[0]
    if direct:
        return None
    next_hop = [item for item in layout_switches if item.get('current_layout') == current_layout
        and item.get('target_layout') == next_layout]
    return next_hop[0] if len(next_hop) == 1 else None


def _literal_key_conflicts_with_qwerty(item: dict[str, Any], *, qwerty_geometry: dict[str, Any]) -> bool:
    """Reject alternate glyphs painted over a QWERTY letter or backspace key."""
    bounds = item.get("bounds")
    anchors = qwerty_geometry.get("anchors")
    if not isinstance(bounds, list) or len(bounds) != 4 or (not isinstance(anchors, dict)):
        return True
    backspace = anchors.get('backspace')
    if not isinstance(backspace, (list, tuple)) or len(backspace) != 2:
        return True
    try:
        profile = qwerty_keyboard_config_from_anchors(anchors)
        left, top, right, bottom = (float(part) for part in bounds)
        backspace_x, backspace_y = (float(part) for part in backspace)
        rows = profile['rows']
        row_y = [float(row['y']) for row in rows]
    except (TypeError, ValueError, WorkflowNotReady):
        return True
    if left <= backspace_x <= right and top <= backspace_y <= bottom:
        return True
    if not isinstance(rows, list) or len(rows) != 3:
        return True
    vertical_pitch = min(row_y[1] - row_y[0], row_y[2] - row_y[1])
    if vertical_pitch <= 0:
        return True
    center_x = (left + right) / 2000.0
    center_y = (top + bottom) / 2000.0
    return any(abs(center_y - float(row['y'])) <= 0.45 * vertical_pitch and any(abs(center_x -
        (float(row['x_start']) + index * float(row['x_step']))) <= 0.48 * float(row['x_step'])
        for index in range(len(str(row.get('keys') or '')))) for row in rows)


def _validated_keyboard_controls(keyboard: dict[str, Any], *, keyboard_bounds: NormalizedBounds | None,
    keyboard_layout: str, keyboard_input_mode: str, keyboard_case_mode: str,
    snapped_anchors: dict[str, list[int]] | None) -> dict[str, Any]:
    """Validate all optional keyboard controls through one shared schema and geometry gate."""

    def control(value: Any, *, field_name: str, fields: set[str], require_visible: bool=False,
        tolerance: float=12) -> dict[str, Any] | None:
        if (keyboard_bounds is None or not isinstance(value, dict) or set(value) != fields
            or (require_visible and value.get('fully_visible') is not True)
            or not _valid_1000_bounds(value.get('bounds'))):
            return None
        try:
            confidence = _audit_confidence(value.get('confidence'), field_name)
        except UISceneError:
            return None
        label = str(value.get('label') or '').strip()
        bounds = tuple(float(part) for part in value['bounds'])
        if not label or confidence < 0.9 or not _bounds_inside(bounds, keyboard_bounds, tolerance=tolerance):
            return None
        return {'label': label, 'bounds': [round(part) for part in bounds], 'confidence': confidence}

    def directional(value: Any, *, field_name: str, modes: frozenset[str],
        label_matches: Callable[[str], bool], current_mode: str | None=None, current_key: str='current_mode',
        target_key: str='target_mode', tolerance: float=12) -> dict[str, Any] | None:
        base = control(value, field_name=field_name,
            fields={'label', 'bounds', 'confidence', current_key, target_key}, tolerance=tolerance)
        if base is None:
            return None
        current, target = value[current_key], value[target_key]
        if (current not in modes or target not in modes or current == target
            or (current_mode is not None and current != current_mode) or not label_matches(base['label'])):
            return None
        return {**base, current_key: current, target_key: target}

    def items(value: Any, *, field_name: str, maximum: int, fields: set[str], item_error: str
        ) -> list[dict[str, Any]]:
        reject_if(not isinstance(value, list) or len(value) > maximum,
            UISceneError(f"{field_name} 必须是最多{maximum}项的数组。"))
        reject_if(value and keyboard_bounds is None, UISceneError(f"{field_name} 必须绑定完整可见键盘。"))
        for item in value:
            reject_if(not isinstance(item, dict) or set(item) != fields, UISceneError(item_error))
        return value

    mode = directional(keyboard.get('mode_switch'), field_name='mode_switch',
        modes=frozenset({'direct_latin', 'chinese_pinyin'}), label_matches=_is_explicit_keyboard_mode_label,
        tolerance=20)
    if mode is not None:
        bounds = tuple(float(part) for part in keyboard['mode_switch']['bounds'])
        width, height = keyboard_bounds[2] - keyboard_bounds[0], keyboard_bounds[3] - keyboard_bounds[1]
        if ((bounds[2] - bounds[0] > 0.35 * width) or (bounds[3] - bounds[1] > 0.3 * height)
            or (keyboard_input_mode != 'unknown' and mode['current_mode'] != keyboard_input_mode)):
            mode = None

    base = control(keyboard.get('backspace_key'), field_name='backspace_key',
        fields={'label', 'bounds', 'confidence', 'fully_visible'}, require_visible=True)
    backspace = None
    if base is not None and re.search('(?:⌫|⌦|退格|删除|backspace|delete)', base['label'], re.IGNORECASE):
        left, top, right, bottom = (float(part) for part in keyboard['backspace_key']['bounds'])
        backspace = {'label': base['label'], 'center': [round((left + right) / 2), round((top + bottom) / 2)],
            'confidence': base['confidence']}

    if snapped_anchors is not None:
        enter = _locally_snapped_keyboard_enter_key(keyboard.get('enter_key'), anchors=snapped_anchors,
            keyboard_bounds=keyboard_bounds)
    else:
        enter = control(keyboard.get('enter_key'), field_name='enter_key',
            fields={'label', 'bounds', 'confidence', 'fully_visible', 'key_action'}, require_visible=True)
        if enter is None or keyboard['enter_key']['key_action'] not in {
            'newline', 'send', 'search', 'done', 'next', 'unknown'}:
            enter = None
        else:
            enter['key_action'] = keyboard['enter_key']['key_action']

    case = directional(keyboard.get('case_switch'), field_name='case_switch',
        modes=frozenset({'lower', 'upper'}), current_mode=keyboard_case_mode,
        label_matches=lambda label: bool(re.search('(?:shift|大小写|大写|小写|⇧|↑|⬆)', label, re.IGNORECASE)))
    if keyboard_layout != 'qwerty' or keyboard_input_mode != 'direct_latin':
        case = None

    literal_keys: list[dict[str, Any]] = []
    literal_items = items(keyboard.get('literal_keys', []), field_name='literal_keys', maximum=8,
        fields={'value', 'label', 'key_kind', 'bounds', 'confidence', 'fully_visible'},
        item_error='literal_key 字段不符合协议。')
    for item in literal_items:
        key_value, label, key_kind = item.get('value'), item.get('label'), item.get('key_kind')
        reject_if(not isinstance(key_value, str) or len(key_value) != 1 or not isinstance(label, str)
            or key_kind not in {'character', 'space'} or not _valid_1000_bounds(item.get('bounds'))
            or not isinstance(item.get('fully_visible'), bool), UISceneError("literal_key 内容无效。"))
        if (key_kind == 'character' and (key_value == ' ' or label != key_value)
            or key_kind == 'space' and (key_value != ' ' or label.strip().casefold() not in {'', 'space', '空格'})):
            # Revoke a mismatched optional glyph without vetoing independent input or layout evidence.
            continue
        bounds = tuple(float(part) for part in item['bounds'])
        confidence = _audit_confidence(item.get('confidence'), 'literal_key')
        if (item['fully_visible'] is not True or confidence < 0.9 or keyboard_bounds is None
            or not _bounds_inside(bounds, keyboard_bounds, tolerance=12)):
            continue
        if key_kind == 'space':
            width, height = keyboard_bounds[2] - keyboard_bounds[0], keyboard_bounds[3] - keyboard_bounds[1]
            if bounds[2] - bounds[0] < 0.18 * width or bounds[1] < keyboard_bounds[1] + 0.55 * height:
                continue
        literal_keys.append({'value': key_value, 'label': label, 'key_kind': key_kind,
            'bounds': [round(part) for part in bounds], 'confidence': confidence})

    layout_switches: list[dict[str, Any]] = []
    layout_items = items(keyboard.get('layout_switches', []), field_name='layout_switches', maximum=4,
        fields={'label', 'bounds', 'confidence', 'current_layout', 'target_layout'},
        item_error='layout_switch 字段不符合协议。')
    patterns = {'numeric': r'(?:123|数字|num)', 'qwerty': r'(?:abc|字母|英文|letters?)',
        'symbol': r'(?:符|sym|[#?+]=?|[.?]123)'}
    for item in layout_items:
        target = item.get('target_layout')
        _audit_confidence(item.get('confidence'), 'layout_switch')
        parsed = directional(item, field_name='layout_switch', modes=frozenset(patterns),
            current_mode=keyboard_layout, current_key='current_layout', target_key='target_layout',
            label_matches=lambda label, pattern=patterns.get(target): bool(pattern and re.search(pattern,
                label.strip().casefold())))
        if parsed is not None:
            layout_switches.append(parsed)
    return {'mode': mode, 'backspace': backspace, 'enter': enter, 'case': case,
        'literal_keys': literal_keys, 'layout_switches': layout_switches}


def _is_explicit_keyboard_mode_label(label: str) -> bool:
    visible = re.sub(r"[\s_\-/]+", "", str(label or "").strip().casefold())
    if not visible:
        return False
    if '中' in visible or '英' in visible:
        return True
    return visible in {'en', 'eng', 'english', '中文', 'chinese', 'abc', 'latin', 'pinyin'}


def _bounds_inside(inner: NormalizedBounds, outer: NormalizedBounds, *,
    tolerance: float) -> bool:
    return inner[0] >= outer[0] - tolerance and inner[1] >= outer[1] - tolerance and (inner[2] <= outer[2] +
        tolerance) and (inner[3] <= outer[3] + tolerance)


def _vertical_overlap_ratio(first: NormalizedBounds, second: NormalizedBounds) -> float:
    overlap = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    smaller = min(first[3] - first[1], second[3] - second[1])
    return overlap / smaller if smaller > 0 else 0.0


def _bounds_overlap_ratio(first: NormalizedBounds, second: NormalizedBounds) -> float:
    """Return how much of ``first`` is covered by ``second``."""

    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    return width * height / first_area if first_area > 0 else 0.0

def _strip_model_authored_local_attestations(payload: dict[str, Any]) -> None:
    """Private controller facts can only be minted by local audit code."""

    elements = payload.get("elements")
    if not isinstance(elements, list):
        return
    for item in elements:
        if not isinstance(item, dict) or not isinstance(item.get('states'), dict):
            continue
        item["states"].pop("independent_geometry_verified", None)
        item["states"].pop("geometry_audit_source", None)
        item["states"].pop("focus_only_input_surface", None)
        states = item["states"]
        if states.get('keyboard_input_mode_switch') is True:
            modes = {"direct_latin", "chinese_pinyin"}
            current_mode = states.get("current_mode")
            target_mode = states.get("target_mode")
            if current_mode not in modes or target_mode not in modes or current_mode == target_mode:
                # Revoke incomplete model switch claims; only local audit may mint a directional switch.
                states.pop("keyboard_input_mode_switch", None)
                states.pop("current_mode", None)
                states.pop("target_mode", None)


def _observation_cache_key(*, device_id: str | None, fingerprint: str, goal_context: dict[str, Any],
    input_lineage: TypedInputLineage | None, post_action_context: post_action_contract.PostActionVisualContext |
    None, available_action_kinds: tuple[str, ...]) -> str | None:
    resolved_device = str(device_id or "").strip()
    if resolved_device.casefold() in {'', 'unbound', 'unknown', 'none', 'null'}:
        return None
    lineage_payload = input_lineage.to_dict() if input_lineage is not None else None
    payload = {'device_id': resolved_device, 'fingerprint': str(fingerprint or '').strip(),
        'goal_context': goal_context, 'input_lineage': lineage_payload,
        'post_action_context': post_action_context.to_dict() if post_action_context is not None else None,
        'available_action_kinds': list(available_action_kinds)}
    return canonical_digest(payload)
