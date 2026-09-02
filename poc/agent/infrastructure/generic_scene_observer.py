"""Local visual-observation adapter for the generic Agent runtime."""

from __future__ import annotations

from agent.domain.validation import NormalizedBounds, reject_if
import json
from functools import lru_cache
from importlib.resources import files
import math
import re
import statistics
import threading
import time
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable

from PIL import Image
from agent.infrastructure.observation_images import (
    consensus_top_edge_obstructions,
    local_frame_fingerprint,
    measure_frame_sharpness,
    measure_local_stability,
)
from agent.domain.foreground_app_identity import ForegroundAppIdentity
from agent.domain.canonical_action_kinds import CANONICAL_ACTION_KINDS
from agent.domain.canonical_action_protocol import (
    CanonicalActionProtocolError,
    normalize_model_step_decision,
)
from agent.infrastructure.qwen_runtime_errors import classify_qwen_error
from agent.infrastructure.robot_controller import WorkflowNotReady, qwerty_keyboard_config_from_anchors
from agent.domain.ui_scene import (
    UI_SCENE_PROTOCOL_VERSION,
    UIElement,
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
import agent.domain.generic_goal as generic_goal_domain

SINGLE_STEP_SCENE_OBSERVER_VERSION = "2026-09-02-single-step-scene-action-finish-v9"
SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION = "2026-09-01-single-step-qwen-action-finish-v6"
INPUT_STRUCTURE_AUDIT_VERSION = "2026-09-01-input-structure-audit-v12"
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

_ACTION_LIKE_WIRE_KEYS = frozenset({'action', 'actions', 'plan', 'plans', 'step', 'steps', 'tap', 'swipe',
    'command', 'shell', 'coordinates', 'next_action', 'execution_plan'})

_KEYBOARD_REQUIRED_FIELDS = frozenset({"visible", "bounds", "layout", "input_mode", "mode_switch"})
_KEYBOARD_OPTIONAL_FIELDS = frozenset({'qwerty_anchors', 'backspace_key', 'enter_key', 'case_mode', 'case_switch',
    'literal_keys', 'layout_switches'})
_INPUT_AUDIT_FIELDS = frozenset({'element_id', 'structure_id', 'bounds', 'fully_visible', 'text', 'placeholder',
    'visible_editable_cues', 'caret_line_index', 'confidence', 'right_button'})


def _hidden_keyboard_payload() -> dict[str, Any]:
    return {'visible': False, 'bounds': None, 'layout': 'unknown', 'input_mode': 'unknown',
        'case_mode': 'unknown', 'qwerty_anchors': None, 'mode_switch': None, 'backspace_key': None,
        'enter_key': None, 'case_switch': None, 'literal_keys': [], 'layout_switches': []}


def _normalize_input_structure_payload(value: Any) -> dict[str, Any]:
    """Fill only fixed wire defaults; never invent input geometry or text."""

    if value is None:
        value = {}
    reject_if(not isinstance(value, Mapping), UISceneError("input_structure必须是对象或null。"))
    allowed = {'protocol_version', 'application_inputs', 'ime_preedit_regions', 'keyboard'}
    reject_if(_contains_action_like_extra(value, allowed), UISceneError("input_structure包含动作或计划字段。"))
    version = value.get('protocol_version', INPUT_STRUCTURE_AUDIT_VERSION)
    reject_if(version != INPUT_STRUCTURE_AUDIT_VERSION, UISceneError("输入结构审计协议版本不匹配。"))
    application_inputs = value.get('application_inputs', [])
    ime_preedit_regions = value.get('ime_preedit_regions', [])
    if not isinstance(application_inputs, list):
        application_inputs = []
    if not isinstance(ime_preedit_regions, list):
        ime_preedit_regions = []
    raw_keyboard = value.get('keyboard')
    if raw_keyboard is None:
        keyboard = _hidden_keyboard_payload()
    elif isinstance(raw_keyboard, Mapping):
        reject_if(_contains_action_like_extra(raw_keyboard,
            _KEYBOARD_REQUIRED_FIELDS | _KEYBOARD_OPTIONAL_FIELDS),
            UISceneError("输入结构审计 keyboard 包含动作或计划字段。"))
        keyboard = _hidden_keyboard_payload()
        keyboard.update({key: raw_keyboard[key] for key in _KEYBOARD_REQUIRED_FIELDS | _KEYBOARD_OPTIONAL_FIELDS
            if key in raw_keyboard})
    else:
        keyboard = _hidden_keyboard_payload()
    return {'protocol_version': INPUT_STRUCTURE_AUDIT_VERSION, 'application_inputs': list(application_inputs),
        'ime_preedit_regions': list(ime_preedit_regions), 'keyboard': keyboard}


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


def _contains_action_like_wire_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any((str(key).strip().casefold() in _ACTION_LIKE_WIRE_KEYS
            or _contains_action_like_wire_key(part) for key, part in value.items()))
    if isinstance(value, list):
        return any(_contains_action_like_wire_key(part) for part in value)
    return False


def _contains_action_like_extra(value: Mapping[str, Any], allowed: Iterable[str]) -> bool:
    allowed_keys = frozenset(allowed)
    return _contains_action_like_wire_key({key: part for key, part in value.items() if key not in allowed_keys})


STAGE_LABELS = {'idle': '空闲', 'checking_stability': '检查画面稳定性', 'waiting_single_step_observation': '等待千问单步完整观察',
    'parsing_single_step_observation': '解析单步完整观察', 'completed': '观察完成', 'failed': '观察安全停止'}


class _SingleStepObserverBase:
    """Shared state and transport for the sole production scene observer."""

    def __init__(self, provider: Any, *,
        qwerty_row_snapper: Callable[[list[Image.Image] | tuple[Image.Image, ...], dict[str, Any]], dict[str,
        list[int]] | None] | None=None) -> None:
        self.provider = provider
        self.qwerty_row_snapper = qwerty_row_snapper
        self.last_raw_response = ""
        self.last_diagnostics: dict[str, Any] = {}
        self._stage_lock = threading.RLock()
        self._current_stage = "idle"
        self._last_stage = "idle"
        self.supports_runtime_action_contract = True
        self.supports_trusted_foreground_identity = True

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
        device_id: str | None=None, available_action_kinds: Iterable[str] | None=None,
        trusted_foreground_identity: ForegroundAppIdentity | None=None) -> UIScene:
        scene, _ = self.observe_with_decision(frames=frames, goal_context=goal_context, device_id=device_id,
            available_action_kinds=available_action_kinds,
            trusted_foreground_identity=trusted_foreground_identity)
        return scene

    def observe_with_decision(self, *, frames: list[Image.Image], goal_context: dict[str, Any] | None=None,
        device_id: str | None=None, available_action_kinds: Iterable[str] | None=None,
        trusted_foreground_identity: ForegroundAppIdentity | None=None) -> tuple[UIScene, dict[str, Any]]:
        self.last_raw_response = ""
        model_identity = public_model_identity(self.provider.status())
        if trusted_foreground_identity is not None:
            trusted_foreground_identity.validate()
            reject_if(not isinstance(device_id, str) or trusted_foreground_identity.device_id != device_id,
                VisionAgentError("Companion 前台 App 身份与本轮观察设备不一致。"))
        self.last_diagnostics = {"vision_model": model_identity,
            "foreground_identity_source": trusted_foreground_identity.source if trusted_foreground_identity else
            "qwen_visual"}
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
            input_structure_required = goal.input_requested
            model_frames = tuple(frames[stable_tail_start:]) if input_structure_required else (frame,)

            request_image_sizes = {_image_request_size(item) for item in model_frames}
            reject_if(len(request_image_sizes) != 1, VisionAgentError("同一步发送给Qwen的稳定帧尺寸不一致，不能建立唯一坐标空间。"))
            request_image_size = next(iter(request_image_sizes))

            prompt = _single_step_observation_prompt(context, include_input_structure=input_structure_required,
                image_count=len(model_frames), request_image_size=request_image_size,
                available_action_kinds=runtime_actions,
                trusted_foreground_identity=trusted_foreground_identity)
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
            model_foreground_app_id = str(scene_payload.get("foreground_app_id")
                or scene_payload.get("app_id") or "unknown").strip()
            if trusted_foreground_identity is not None:
                scene_payload["foreground_app_id"] = trusted_foreground_identity.package_name
                scene_payload["app_id"] = trusted_foreground_identity.package_name
            single_step_input_surface = _single_step_input_surface_attestation(scene_payload,
                input_structure=envelope["input_structure"], goal_context=context) if input_structure_required else None
            obstructions = consensus_top_edge_obstructions(frames[stable_tail_start:])
            referenced_element_ids = _decision_element_ids(model_decision)
            scene = _parse_scene(json.dumps(scene_payload, ensure_ascii=False, separators=(',', ':')),
                fingerprint=fingerprint, camera_layout_orientation=_camera_layout_orientation(frame),
                strict_element_ids=referenced_element_ids)

            if input_structure_required:
                input_payload = envelope["input_structure"]
                assert isinstance(input_payload, dict)
                scene = _apply_input_structure_audit(scene, json.dumps(input_payload,
                    ensure_ascii=False, separators=(',', ':')), fingerprint=fingerprint, goal_context=context,
                    qwerty_row_snapper=self.qwerty_row_snapper, qwerty_row_frames=frames[stable_tail_start:],
                    single_step_input_surface=single_step_input_surface)

            reject_if(not scene.stable, VisionAgentError("页面仍在变化，不能建立可信候选。"))

            for selected_ref in referenced_element_ids:
                matches = [item for item in scene.elements if item.element_id == selected_ref]
                reject_if(len(matches) != 1,
                    VisionAgentError(f"当前Qwen决策引用的元素不唯一或不可执行：{selected_ref}"))

            self.last_diagnostics = {'observer_version': SINGLE_STEP_SCENE_OBSERVER_VERSION,
                'vision_model': public_model_identity(self.provider.status()),
                'strategy': 'single_step_current_scene_observation',
                'protocol_version': SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION, 'model_calls': 1,
                'online_stages': ['single_step_observation'],
                'input_structure_in_same_response': input_structure_required, 'remote_retry_used': False,
                'selected_frame_index': selected_frame_index,
                'stable_tail_start_index': stable_tail_start, 'local_stability': stability.to_dict(),
                'frame_sharpness_scores': [round(value, 3) for value in sharpness_scores],
                'frame_size': list(frame.size), 'request_image_size': list(request_image_size),
                'coordinate_normalization': envelope.get('coordinate_normalization'), 'fingerprint': fingerprint,
                'element_count': len(scene.elements), 'decision_status': model_decision['status'],
                'foreground_identity_source': trusted_foreground_identity.source
                if trusted_foreground_identity else 'qwen_visual',
                'trusted_foreground_app_id': trusted_foreground_identity.package_name
                if trusted_foreground_identity else None,
                'model_foreground_app_id': model_foreground_app_id,
                'available_action_kinds': list(runtime_actions),
                'local_visual_obstructions': [item.to_dict() for item in obstructions],
                'model_call_elapsed_seconds': [call_elapsed],
                'model_call_token_budgets': [SINGLE_STEP_OUTPUT_TOKENS],
                'elapsed_seconds': round(time.perf_counter() - started, 3)}
            self._set_stage("completed")
            return scene, model_decision
        except Exception as exc:
            failed_stage = self.status()["last_stage"]
            self._set_stage("failed")
            self.last_diagnostics = {'observer_version': SINGLE_STEP_SCENE_OBSERVER_VERSION,
                'vision_model': public_model_identity(self.provider.status()),
                'strategy': 'single_step_current_scene_observation',
                'protocol_version': SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION, 'model_calls': model_calls,
                'online_stages': ['single_step_observation'] if model_calls else [],
                'input_structure_in_same_response': input_structure_required, 'remote_retry_used': False,
                'failed_stage': failed_stage, 'fingerprint': fingerprint, 'error': str(exc),
                'foreground_identity_source': trusted_foreground_identity.source
                if trusted_foreground_identity else 'qwen_visual',
                'trusted_foreground_app_id': trusted_foreground_identity.package_name
                if trusted_foreground_identity else None,
                'error_type': classify_qwen_error(exc, raw_response=self.last_raw_response),
                'safe_stop_reason': '单次模型输出未建立完整可信观察；没有发起第二次Qwen请求，控制器与机械臂均未执行。',
                'raw_response_length': len(self.last_raw_response),
                'raw_response_excerpt': self.last_raw_response[:1000],
                'elapsed_seconds': round(time.perf_counter() - started, 3)}
            raise
        finally:
            self._set_stage("idle")


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
    image_count: int, request_image_size: tuple[int, int],
    available_action_kinds: tuple[str, ...],
    trusted_foreground_identity: ForegroundAppIdentity | None=None) -> str:
    request_width, request_height = request_image_size
    scene_contract = _compact_prompt(context, wire_height=request_height,
        input_structure_is_value_authority=include_input_structure,
        trusted_foreground_identity=trusted_foreground_identity)
    if include_input_structure:
        input_contract = _input_structure_audit_prompt(context, wire_height=request_height)
        input_rule = ("input_structure报告下面INPUT CONTRACT中当前可见的最小事实；空输入只需合法bounds和"
            "text=\"\"，无关可选字段、整套键盘几何和另一个scene输入框均不必补齐。")
    else:
        input_contract = "本轮子目标与文字输入无关。"
        input_rule = "input_structure必须为null，不得额外枚举键盘或输入结构。"
    temporal_rule = (f"共有{image_count}张同一稳定手机画面的时间对齐帧。只把它们合并为一个当前状态；"
        "闪烁光标可从任一帧读取，其他瞬态不得合并。" if image_count > 1 else "只有一张当前稳定手机画面。")
    return _render_prompt("single_step_observation.txt", SCENE_CONTRACT=scene_contract,
        INPUT_CONTRACT=input_contract, TEMPORAL_RULE=temporal_rule,
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


def _decision_element_ids(decision: Mapping[str, Any]) -> tuple[str, ...]:
    """Return every scene element that this same-envelope decision makes authoritative."""

    values = [str(decision.get(name) or '').strip() for name in (
        'element_id', 'source_element_id', 'destination_element_id') if decision.get(name)]
    values.extend(str(item).split(':', 1)[1].strip() for item in decision.get('evidence_refs', ())
        if isinstance(item, str) and item.startswith('element:') and str(item).split(':', 1)[1].strip())
    return tuple(dict.fromkeys(values))


def _normalize_single_step_wire_coordinates(payload: dict[str, Any], *, request_image_size: tuple[int,
    int], decision: Mapping[str, Any]) -> dict[str, Any]:
    """Convert the sole request-bound axis grid to canonical 0..1000."""
    coordinate_space = payload.get("coordinate_space")
    request_width, request_height = request_image_size
    expected = {"kind": "axis_grid", "width": 1000, "height": request_height}
    reject_if(coordinate_space != expected, UISceneError("单步观察必须声明唯一coordinate_space。"))
    reject_if(request_width <= 0 or request_height <= 0, UISceneError("本轮Qwen请求图片尺寸无效。"))

    def normalized_bounds(value: Any, *, selected: bool=False) -> list[int] | None:
        valid_shape = bool(isinstance(value, (list, tuple)) and len(value) == 4
            and all(isinstance(part, (int, float)) and not isinstance(part, bool) for part in value))
        if not valid_shape:
            reject_if(selected, UISceneError("已选目标的bounds格式无效。"))
            return None
        left, top, right, bottom = (float(part) for part in value)
        valid_extent = bool(all(math.isfinite(part) for part in (left, top, right, bottom))
            and 0 <= left < right <= 1000 and 0 <= top < bottom <= request_height)
        if not valid_extent:
            reject_if(selected, UISceneError("已选目标的bounds超出声明的axis_grid。"))
            return None
        result = [round(left), round(top * 1000 / request_height), round(right),
            round(bottom * 1000 / request_height)]
        if not _valid_1000_bounds(result):
            reject_if(selected, UISceneError("已选目标的axis_grid换算后bounds退化。"))
            return None
        return result

    def normalize_optional_control(value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        result = dict(value)
        bounds = normalized_bounds(result.get('bounds'))
        if bounds is None:
            return None
        result['bounds'] = bounds
        return result

    scene = payload.get('scene')
    reject_if(_contains_action_like_wire_key(scene), UISceneError("单步观察scene包含动作或计划字段。"))
    selected_ids = set(_decision_element_ids(decision))
    if isinstance(scene, dict):
        raw_elements = scene.get('elements')
        normalized_elements: list[dict[str, Any]] = []
        selected_counts: dict[str, int] = {}
        if isinstance(raw_elements, list):
            for item in raw_elements:
                if not isinstance(item, dict):
                    continue
                element_id = str(item.get('element_id') or '').strip()
                selected = element_id in selected_ids
                if selected:
                    selected_counts[element_id] = selected_counts.get(element_id, 0) + 1
                    reject_if(selected_counts[element_id] > 1,
                        UISceneError(f"已选目标的element_id不唯一：{element_id}"))
                bounds = normalized_bounds(item.get('bounds'), selected=selected)
                if bounds is None:
                    continue
                normalized = dict(item)
                normalized['bounds'] = bounds
                normalized_elements.append(normalized)
        scene['elements'] = normalized_elements

    input_structure = payload.get('input_structure')
    reject_if(_contains_action_like_wire_key(input_structure),
        UISceneError("单步观察input_structure包含动作或计划字段。"))
    if isinstance(input_structure, dict):
        raw_inputs = input_structure.get('application_inputs')
        normalized_inputs: list[dict[str, Any]] = []
        invalid_inputs = 0
        if isinstance(raw_inputs, list):
            for item in raw_inputs:
                if not isinstance(item, dict):
                    invalid_inputs += 1
                    continue
                bounds = normalized_bounds(item.get('bounds'))
                if bounds is None:
                    invalid_inputs += 1
                    continue
                normalized = dict(item)
                normalized['bounds'] = bounds
                right_button = normalize_optional_control(normalized.get('right_button'))
                normalized['right_button'] = right_button
                normalized_inputs.append(normalized)
        selected_element_id = str(decision.get('element_id') or '').strip()
        selected_scene_input = any(isinstance(item, dict)
            and str(item.get('element_id') or '').strip() == selected_element_id
            and str(item.get('role') or '').strip() == 'input' for item in raw_elements or [])
        selected_audit_input = any(isinstance(item, dict)
            and str(item.get('element_id') or '').strip() == selected_element_id for item in raw_inputs or [])
        selected_input = bool(decision.get('status') == 'action' and selected_element_id
            and (selected_element_id == 'local_audited_input_1' or selected_scene_input or selected_audit_input))
        reject_if(selected_input and not normalized_inputs and invalid_inputs > 0,
            UISceneError("已选输入框的bounds无效。"))
        input_structure['application_inputs'] = normalized_inputs

        raw_preedits = input_structure.get('ime_preedit_regions')
        normalized_preedits: list[dict[str, Any]] = []
        if isinstance(raw_preedits, list):
            for region in raw_preedits:
                if not isinstance(region, dict):
                    continue
                bounds = normalized_bounds(region.get('bounds'))
                if bounds is None:
                    continue
                normalized_region = dict(region)
                normalized_region['bounds'] = bounds
                raw_candidates = region.get('candidates')
                normalized_region['candidates'] = [item for item in (
                    normalize_optional_control(candidate) for candidate in (
                        raw_candidates if isinstance(raw_candidates, list) else [])) if item is not None]
                normalized_preedits.append(normalized_region)
        input_structure['ime_preedit_regions'] = normalized_preedits

        raw_keyboard = input_structure.get('keyboard')
        if isinstance(raw_keyboard, dict):
            keyboard = dict(raw_keyboard)
            keyboard['bounds'] = normalized_bounds(keyboard.get('bounds'))
            anchors = keyboard.get('qwerty_anchors')
            normalized_anchors: dict[str, list[int]] | None = None
            if isinstance(anchors, dict):
                candidate_anchors: dict[str, list[int]] = {}
                for key, point in anchors.items():
                    valid_point = bool(isinstance(point, (list, tuple)) and len(point) == 2
                        and all(isinstance(part, (int, float)) and not isinstance(part, bool) for part in point))
                    if not valid_point:
                        candidate_anchors = {}
                        break
                    x, y = (float(part) for part in point)
                    if not (math.isfinite(x) and math.isfinite(y)
                        and 0 <= x <= 1000 and 0 <= y <= request_height):
                        candidate_anchors = {}
                        break
                    candidate_anchors[str(key)] = [round(x), round(y * 1000 / request_height)]
                if candidate_anchors:
                    normalized_anchors = candidate_anchors
            keyboard['qwerty_anchors'] = normalized_anchors
            for name in ('mode_switch', 'backspace_key', 'enter_key', 'case_switch'):
                keyboard[name] = normalize_optional_control(keyboard.get(name))
            for name in ('literal_keys', 'layout_switches'):
                raw_items = keyboard.get(name)
                keyboard[name] = [item for item in (normalize_optional_control(value) for value in (
                    raw_items if isinstance(raw_items, list) else [])) if item is not None]
            input_structure['keyboard'] = keyboard
    payload.pop("coordinate_space", None)
    return {'wire_kind': 'axis_grid', 'wire_extent': [1000, request_height], 'request_image_size': [request_width,
        request_height], 'canonical_extent': [1000, 1000], 'applied': True}


def _parse_single_step_observation_envelope(raw: str, *, input_structure_required: bool, request_image_size: tuple[int,
    int]) -> dict[str, Any]:
    """Parse one current-scene response without remote repair or resampling."""

    try:
        payload = _extract_json_object(raw, reject_duplicate_keys=True)
        required = {'coordinate_space', 'scene', 'decision'}
        missing = sorted(required - set(payload))
        reject_if(bool(missing), UISceneError("单步观察封装结构无效；缺少字段：" + ", ".join(missing)))
        reject_if(_contains_action_like_extra(payload, required | {'protocol_version', 'input_structure'}),
            UISceneError("单步观察封装包含动作或计划字段。"))
        version = payload.get('protocol_version', SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION)
        reject_if(version != SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION, UISceneError("单步观察协议版本不匹配。"))
        reject_if(not isinstance(payload['scene'], dict), UISceneError("单步观察scene必须是对象。"))
        if input_structure_required:
            payload['input_structure'] = _normalize_input_structure_payload(payload.get('input_structure'))
        else:
            # Irrelevant optional input facts cannot veto a non-input action.
            payload['input_structure'] = None
        decision = normalize_model_step_decision(payload['decision'])
        coordinate_normalization = _normalize_single_step_wire_coordinates(payload,
            request_image_size=request_image_size, decision=decision)
        return {'scene': payload['scene'], 'input_structure': payload['input_structure'],
            'decision': decision, 'coordinate_normalization': coordinate_normalization}
    except (CanonicalActionProtocolError, UISceneError, ValueError, TypeError) as exc:
        raise VisionAgentError(f"单步完整观察结果不符合协议：{exc}") from exc


def _compact_prompt(context: dict[str, Any], *, wire_height: int=1000,
    input_structure_is_value_authority: bool=False,
    trusted_foreground_identity: ForegroundAppIdentity | None=None) -> str:
    context = _goal_view(context).observation_context
    if _goal_view(context).mode_switch_requested:
        keyboard_switch_rule = (" 当前子目标明确要求切换键盘输入模式；本轮快速观察不得在elements中报告或定位"
            "任何模式切换键。后续独立全帧输入结构审计是模式、方向和模式键几何的唯一权威。"
            "普通输入框和键盘可见事实仍可报告，但不得据此建议动作。")
    else:
        keyboard_switch_rule = KEYBOARD_MODE_SWITCH_OBSERVATION_RULE
    input_rule = (("本轮input_structure.application_inputs.text是应用输入正文空/非空事实的"
        "唯一视觉权威。scene中的role=input只是可选页面上下文，不必与input_structure重复出现或"
        "提供正文、占位符、光标、证据、置信度阈值或几何重合；不得用scene事实否决input_structure。"
        "键盘模式、IME和可执行输入几何只在同一响应的input_structure中按当前动作需要报告。"
        + LOCAL_TEXT_CLEAR_OBSERVATION_RULE) if input_structure_is_value_authority else
        INPUT_VALUE_AND_MODE_OBSERVATION_RULE + keyboard_switch_rule + LOCAL_TEXT_CLEAR_OBSERVATION_RULE)
    if trusted_foreground_identity is None:
        foreground_identity_rule = ("桌面写 launcher；系统最近任务页面必须写foreground_app_id=system、"
            "screen_id=system_recent_tasks；不确定写 unknown。不得把目标App当成当前App，也不得把"
            "current_foreground、current_app、foreground_app、target_app 或 active_app 等引用占位符"
            "写成foreground_app_id；该字段只能来自当前画面的视觉身份。")
    else:
        trusted_foreground_identity.validate()
        foreground_identity_rule = ("本地已通过配对签名和Android系统接口确定当前前台包名为"
            f"{trusted_foreground_identity.package_name}（source={trusted_foreground_identity.source}）。"
            "foreground_app_id必须逐字写该包名；不得根据JPEG、目标App、嵌入内容或页面文字改写、"
            "覆盖或否决。你仍须仅根据当前JPEG判断screen_id、控件和页面语义。")
    return _render_prompt("compact_scene.txt", CONTEXT=json.dumps(context, ensure_ascii=False, separators=(',', ':')),
        INPUT_RULE=input_rule, MAX_ELEMENTS=str(MAX_COMPACT_ELEMENTS), WIRE_HEIGHT=str(wire_height),
        SCENE_PROTOCOL=UI_SCENE_PROTOCOL_VERSION, FOREGROUND_IDENTITY_RULE=foreground_identity_rule)


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


def _input_structure_audit_prompt(context: dict[str, Any], *, wire_height: int=1000) -> str:
    context = _goal_view(context).observation_context
    goal = _goal_view(context)
    keyboard_min_height = max(1, round(180 * wire_height / 1000))
    literal_targets = _input_audit_literal_key_targets(context)
    literal_example = []
    if literal_targets:
        value = literal_targets[0]
        literal_example = [{'value': value, 'label': 'Space' if value == ' ' else value,
            'key_kind': 'space' if value == ' ' else 'character', 'bounds': [0, 0, 1000, wire_height],
            'confidence': 0.0, 'fully_visible': True}]
    field_id, field_label, multiline = goal.field
    return _render_prompt("input_structure_audit.txt",
        CONTEXT=json.dumps(context, ensure_ascii=False, separators=(',', ':')),
        FIELD_ID=json.dumps(field_id, ensure_ascii=False), FIELD_LABEL=json.dumps(field_label, ensure_ascii=False),
        TARGET_ONLY_CLEAR=str(goal.target_only).lower(), ENTER_REQUIRED='false',
        MULTILINE=str(multiline).lower(), LITERAL_TARGETS=json.dumps(literal_targets, ensure_ascii=False,
        separators=(',', ':')), LITERAL_KEYS_EXAMPLE=json.dumps(literal_example, ensure_ascii=False,
        separators=(',', ':')), WIRE_HEIGHT=str(wire_height), KEYBOARD_MIN_HEIGHT=str(keyboard_min_height),
        AUDIT_VERSION=INPUT_STRUCTURE_AUDIT_VERSION)


def _parse_scene(raw: str, *, fingerprint: str, camera_layout_orientation: str | None=None,
    strict_element_ids: Iterable[str]=()) -> UIScene:
    """Parse the current scene, revoking only malformed optional model facts.

    The same-envelope decision makes its referenced elements required facts.  A
    malformed referenced element is therefore a contract error, while an
    unrelated malformed hint is simply omitted and cannot veto the selected
    action or finish evidence.
    """

    try:
        payload = _extract_json_object(raw)
        if 'system_ui' not in payload:
            payload['system_ui'] = {}
        alignment = payload.get('camera_alignment')
        if not isinstance(alignment, dict):
            alignment = {}
        else:
            alignment = dict(alignment)
        if camera_layout_orientation is not None:
            alignment['camera_layout_orientation'] = camera_layout_orientation
        payload['camera_alignment'] = alignment
        _drop_forbidden_camera_alignment_evidence(payload)
        _strip_model_authored_local_attestations(payload)
        required = frozenset(str(item or '').strip() for item in strict_element_ids if str(item or '').strip())
        raw_elements = payload.get('elements')
        if isinstance(raw_elements, list):
            retained: list[dict[str, Any]] = []
            required_counts: dict[str, int] = {}
            for raw_element in raw_elements:
                if not isinstance(raw_element, dict):
                    continue
                element_id = str(raw_element.get('element_id') or '').strip()
                if element_id in required:
                    required_counts[element_id] = required_counts.get(element_id, 0) + 1
                    reject_if(required_counts[element_id] > 1,
                        UISceneError(f"Qwen决策引用的element_id不唯一：{element_id}"))
                    retained.append(_required_scene_element(raw_element))
                    continue
                try:
                    UIElement.from_dict(raw_element, coordinate_scale=1000.0)
                except (UISceneError, TypeError, ValueError):
                    continue
                retained.append(raw_element)
            payload['elements'] = retained
        return UIScene.from_dict(payload, coordinate_scale=1000.0, stable_override=True,
            fingerprint_override=fingerprint)
    except (UISceneError, ValueError, TypeError) as exc:
        raise VisionAgentError(f"通用页面观察结果不符合协议：{exc}") from exc


def _required_scene_element(raw_element: dict[str, Any]) -> dict[str, Any]:
    """Keep a selected element's hard identity and revoke malformed optional facts."""

    core = {
        'element_id': str(raw_element.get('element_id') or '').strip(),
        'role': str(raw_element.get('role') or 'unknown').strip(),
        'meaning': str(raw_element.get('meaning') or '').strip(),
        'bounds': raw_element.get('bounds'),
        'confidence': _diagnostic_confidence(raw_element.get('confidence')),
        'label': str(raw_element.get('label') or '').strip()[:200],
        'states': {},
        'evidence': [item.strip()[:200] for item in raw_element.get('evidence', [])
            if isinstance(item, str) and item.strip()]
            if isinstance(raw_element.get('evidence'), (list, tuple)) else [],
    }
    # The selected element must always have a legal id/role/meaning/bounds.
    UIElement.from_dict(core, coordinate_scale=1000.0)
    states = raw_element.get('states')
    if isinstance(states, dict):
        enriched = dict(core)
        enriched['states'] = dict(states)
        try:
            UIElement.from_dict(enriched, coordinate_scale=1000.0)
        except (UISceneError, TypeError, ValueError):
            return core
        return enriched
    return core


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
    if len(retained) != len(evidence):
        alignment["evidence"] = retained
        if not retained:
            # HUD/coordinate text cannot prove rotation; revoke only that optional fact.
            alignment["phone_content_rotation"] = "unknown"
            alignment["confidence"] = 0.0


def _goal_view(context: dict[str, Any]) -> generic_goal_domain.ActiveVisualGoal:
    return generic_goal_domain.ActiveVisualGoal.from_context(context)


def _valid_1000_bounds(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    if not all((isinstance(part, (int, float)) and (not isinstance(part, bool)) for part in value)):
        return False
    left, top, right, bottom = (float(part) for part in value)
    return 0 <= left < right <= 1000 and 0 <= top < bottom <= 1000


def _single_step_input_surface_attestation(payload: dict[str, Any], *, input_structure: Any,
    goal_context: dict[str, Any]) -> dict[str, Any] | None:
    """Keep one same-envelope input-surface fact without text or executable geometry authority."""

    goal = _goal_view(goal_context)
    if not (goal.transaction_text or goal.target_only):
        return None
    elements = payload.get("elements")
    if not isinstance(elements, list):
        return None
    candidates: list[dict[str, Any]] = []
    for item in elements:
        if (not isinstance(item, dict) or _contains_action_like_wire_key(item)
            or str(item.get('role') or '').strip() != 'input'):
            continue
        states = item.get("states") or {}
        evidence = item.get("evidence") or []
        raw_bounds = item.get("bounds")
        if (not isinstance(item.get('element_id'), str) or (not item['element_id'].strip())
            or str(item['element_id']).startswith('local_audited_') or (not isinstance(states, dict))
            or (not _valid_1000_bounds(raw_bounds))):
            continue
        bounds = [float(value) for value in raw_bounds]
        if max(bounds) <= 1.0:
            bounds = [value * 1000.0 for value in bounds]
        candidates.append({'element_id': item['element_id'].strip(),
            'meaning': str(item.get('meaning') or 'application_text_input').strip(),
            'label': str(item.get('label') or '').strip()[:200], 'bounds': bounds,
            'confidence': 1.0, 'evidence': tuple(str(value).strip()[:200] for value in evidence
            if isinstance(value, str) and value.strip()) if isinstance(evidence, list) else ()})
    requested_ids: set[str] = set()
    if isinstance(input_structure, Mapping):
        application_inputs = input_structure.get('application_inputs')
        if isinstance(application_inputs, list):
            requested_ids = {str(item.get('element_id') or '').strip() for item in application_inputs
                if isinstance(item, Mapping) and str(item.get('element_id') or '').strip()}
    explicitly_bound = [item for item in candidates if item['element_id'] in requested_ids]
    if len(explicitly_bound) == 1:
        return explicitly_bound[0]
    return candidates[0] if len(candidates) == 1 else None


def _focus_only_compact_input_surface(attestation: dict[str, Any] | None, *, keyboard_visible: bool) -> dict[str,
    Any] | None:
    """Sanitize one compact input into focus-only authority without copying its transcription."""

    if keyboard_visible or not isinstance(attestation, dict):
        return None
    element_id = str(attestation.get("element_id") or "").strip()
    meaning = str(attestation.get("meaning") or "application_text_input").strip()
    label = str(attestation.get("label") or "").strip()[:200]
    bounds = attestation.get("bounds")
    evidence = attestation.get("evidence")
    if (not element_id or element_id.startswith('local_audited_') or (not meaning) or (not isinstance(bounds, (list,
        tuple))) or (len(bounds) != 4) or any((isinstance(value, bool) or not isinstance(value, (int,
        float)) or (not 0 <= float(value) <= 1000) for value in bounds))):
        return None
    return {'element_id': element_id, 'role': 'input', 'meaning': meaning,
        'bounds': [float(value) / 1000.0 for value in bounds], 'confidence': 1.0, 'label': label,
        'states': {'goal_relevant': True, 'fully_visible': True, 'focus_only_input_surface': True},
        'evidence': [str(item).strip()[:200] for item in evidence if isinstance(item, str)
        and item.strip()] if isinstance(evidence, (list, tuple)) else []}


def _projected_input_element_id(attestation: dict[str, Any] | None) -> str:
    """Reuse the one same-response scene identity; mint locally only when scene omitted it."""

    if isinstance(attestation, dict):
        element_id = str(attestation.get('element_id') or '').strip()
        if element_id and not element_id.startswith('local_audited_'):
            return element_id
    return 'local_audited_input_1'


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
    target_field_label: str, input_element_id: str) -> dict[str, Any]:
    return _audited_element('next_field_key', 'input_next_field_key', key, states={'goal_relevant': True,
        'input_next_field_key': True, 'key_action': 'next', 'source_input_field_id': source_field_id,
        'target_input_field_id': target_field_id, 'target_input_field_label': target_field_label,
        'input_element_id': input_element_id}, evidence='唯一聚焦typed字段与完整Next键绑定下一依赖字段')


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
        if (not isinstance(region, dict) or _contains_action_like_wire_key(region)
            or (not _valid_1000_bounds(region.get('bounds'))) or not isinstance(region.get('text', ''), str)):
            continue
        confidence = _diagnostic_confidence(region.get("confidence"))
        bounds = tuple(float(value) for value in region["bounds"])
        candidates = []
        raw_candidates = region.get("candidates", [])
        if isinstance(raw_candidates, list):
            for item in raw_candidates[:8]:
                if (not isinstance(item, dict) or _contains_action_like_wire_key(item)
                    or item.get('fully_visible') is False or (not _valid_1000_bounds(item.get('bounds')))):
                    continue
                text = str(item.get("text") or "").strip()
                item_confidence = _diagnostic_confidence(item.get("confidence"))
                candidate_bounds = tuple(float(value) for value in item["bounds"])
                if (text and len(text) <= 20
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


def _optional_audit_strings(value: Any, *, name: str, limit: int, allow_empty: bool) -> tuple[str, ...]:
    """Project optional diagnostics without letting their shape veto valid input geometry."""

    try:
        return _audit_strings(value, name=name, limit=limit, allow_empty=allow_empty)
    except UISceneError:
        return ()


def _parse_audited_keyboard(value: Any, *, qwerty_row_snapper: Callable[[list[Image.Image] | tuple[Image.Image, ...],
    dict[str, Any]], dict[str, list[int]] | None] | None, qwerty_row_frames: list[Image.Image] | tuple[Image.Image,
    ...] | None) -> _AuditedKeyboard:
    keyboard = _hidden_keyboard_payload()
    if isinstance(value, dict):
        keyboard.update({key: value[key] for key in _KEYBOARD_REQUIRED_FIELDS | _KEYBOARD_OPTIONAL_FIELDS
            if key in value})
    visible = keyboard["visible"] if isinstance(keyboard["visible"], bool) else False
    layout = keyboard["layout"] if keyboard["layout"] in {'qwerty', 'numeric', 'symbol', 'unknown'} else 'unknown'
    input_mode = keyboard["input_mode"] if keyboard["input_mode"] in {'direct_latin', 'chinese_pinyin',
        'unknown'} else 'unknown'
    case_mode = keyboard.get("case_mode", "unknown")
    if case_mode not in {'lower', 'upper', 'unknown'}:
        case_mode = 'unknown'
    keyboard.update(visible=visible, layout=layout, input_mode=input_mode, case_mode=case_mode)

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
    elif not visible:
        keyboard.update(bounds=None, mode_switch=None, backspace_key=None, enter_key=None, case_switch=None,
            literal_keys=[], layout_switches=[])
        keyboard.pop("qwerty_anchors", None)

    application_bounds = bounds
    if _valid_1000_bounds(keyboard.get('bounds')):
        reported = tuple(float(part) for part in keyboard["bounds"])
        if reported[2] - reported[0] >= 300:
            application_bounds = reported
    return _AuditedKeyboard(keyboard, visible, layout, input_mode, case_mode, bounds, application_bounds, raw_anchors,
        snapped_anchors)


def _label_count(labels: Iterable[str], expected: str) -> int:
    folded = expected.casefold()
    return sum(label.casefold() == folded for label in labels)


def _collect_audited_input_matches(application_inputs: list[Any]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for item in application_inputs:
        if not isinstance(item, dict):
            continue
        reject_if(_contains_action_like_extra(item, _INPUT_AUDIT_FIELDS | {'field_labels'}),
            UISceneError("应用输入结构包含动作或计划字段。"))
        if not _valid_1000_bounds(item.get('bounds')):
            continue
        text = item.get('text', '')
        if not isinstance(text, str) or len(text) > 4000:
            continue
        cues = list(_optional_audit_strings(item.get('visible_editable_cues', []),
            name='visible_editable_cues', limit=4, allow_empty=True))
        labels = _optional_audit_strings(item.get('field_labels', []), name='field_labels', limit=6,
            allow_empty=False)
        caret = item.get("caret_line_index")
        if caret is not None and (isinstance(caret, bool) or not isinstance(caret, int) or not 0 <= caret <= 30):
            caret = None
        placeholder = item.get("placeholder")
        placeholder = placeholder.strip()[:200] if isinstance(placeholder, str) else ""
        element_id = item.get('element_id')
        element_id = element_id.strip()[:200] if isinstance(element_id, str) else ''
        bounds = tuple(float(part) for part in item["bounds"])
        button = _optional_right_button(item.get("right_button"), input_bounds=bounds)
        input_bounds = [round(part) for part in bounds]
        if button is not None:
            input_bounds[2] = button["bounds"][0]
        if input_bounds[2] <= input_bounds[0]:
            continue
        match = {'element_id': element_id, 'text': text, 'placeholder': placeholder,
            'visible_editable_cues': cues, 'caret_line_index': caret,
            'field_labels': labels, 'input_bounds': input_bounds, 'right_button': button,
            'confidence': 1.0}
        matches.append(match)
    return matches


def _unique_audited_input(matches: Iterable[dict[str, Any]], *, label: str='', text: str | None=None) -> dict[str,
    Any] | None:
    selected = [item for item in matches if text is None or item['text'] == text]
    if label:
        labeled = [item for item in selected if _label_count(item['field_labels'], label) == 1]
        if len(labeled) == 1:
            return labeled[0]
        if labeled:
            return None
    return selected[0] if len(selected) == 1 else None


_INPUT_FOCUS_CUE_MARKERS = ('caret', 'cursor', '光标', '插入符')
_NEGATED_INPUT_FOCUS_CUE = re.compile(
    r'\b(?:no|not|without|absent|missing|hidden)\b.{0,24}\b(?:caret|cursor)\b|'
    r'\b(?:caret|cursor)\b.{0,24}\b(?:not\s+visible|hidden|absent|missing)\b|'
    r'(?:无|没有|未显示|未出现|未检测到).{0,12}(?:光标|插入符)|'
    r'(?:光标|插入符).{0,12}(?:不可见|隐藏|不存在|缺失)', re.IGNORECASE)


def _audited_input_has_focus_cue(audited_input: dict[str, Any]) -> bool:
    """Normalize only affirmative caret facts from the one input audit response."""

    caret = audited_input.get('caret_line_index')
    if isinstance(caret, int) and not isinstance(caret, bool) and 0 <= caret <= 30:
        return True
    cues = audited_input.get('visible_editable_cues')
    if not isinstance(cues, list):
        return False
    for cue in cues:
        if not isinstance(cue, str):
            continue
        normalized = cue.strip().casefold()
        if normalized == '|':
            return True
        if (not _NEGATED_INPUT_FOCUS_CUE.search(normalized)
            and any(marker in normalized for marker in _INPUT_FOCUS_CUE_MARKERS)):
            return True
    return False


def _append_audited_input_element(elements: list[dict[str, Any]], audited_input: dict[str, Any], *, field_id: str,
    field_label: str, multiline: bool, active_clear_goal: bool, keyboard: _AuditedKeyboard,
    controls: _AuditedInputControls, input_element_id: str='local_audited_input_1') -> None:
    step = controls.step
    auxiliary = any(getattr(controls, key) is not None for key in ("candidate", "literal", "enter", "layout", "case"))
    derived_goal_relevant = bool(not controls.switch_is_goal and controls.next_field is None and (not auxiliary))
    states: dict[str, Any] = {'goal_relevant': derived_goal_relevant,
        'fully_visible': True,
        'value': audited_input['text']}
    if field_id:
        states["input_field_id"] = field_id
        if controls.next_field is None:
            states["input_multiline"] = multiline
    if field_label:
        states["input_field_label"] = field_label
    for key in ('local_caret_line_index', 'clear_extra_delete_units'):
        value = audited_input.get(key)
        if value is True or (isinstance(value, int) and (not isinstance(value, bool)) and (value > 0)):
            states[key] = value
    if not keyboard.visible:
        states["soft_keyboard_visible"] = False
    if audited_input['placeholder']:
        states["placeholder"] = audited_input["placeholder"]
    if keyboard.visible or _audited_input_has_focus_cue(audited_input):
        states["focused"] = True
    if keyboard.visible:
        states.update(keyboard_layout=keyboard.layout, keyboard_input_mode=keyboard.input_mode,
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

    evidence = list(dict.fromkeys((*audited_input['field_labels'], *audited_input['visible_editable_cues'])))
    text = audited_input["text"]
    if text:
        evidence.insert(0, f"应用输入框当前文字：{text}")
        if isinstance(audited_input.get('local_caret_line_index'), int):
            evidence.append(f'模型确认可见光标后，本地校准像素定位光标视觉行：{audited_input['local_caret_line_index']}')
    elif audited_input['placeholder']:
        evidence.insert(0, f"应用输入框为空，占位提示：{audited_input['placeholder']}")
    if controls.clearable_preedit:
        evidence.append("唯一相邻输入法预编辑串已绑定当前typed输入框，可清理文字：" f"{controls.clearable_preedit}")
    if not keyboard.visible:
        evidence.append(AUDITED_SOFT_KEYBOARD_HIDDEN_EVIDENCE)
    audited_element = _audited_element('input', 'application_text_input', audited_input, role='input',
        label=text or audited_input['placeholder'], bounds_key='input_bounds', states=states, evidence=evidence)
    audited_element['element_id'] = input_element_id
    elements.append(audited_element)
    if audited_input['right_button'] is not None:
        elements.append(_audited_element('adjacent_button', 'adjacent_input_utility', audited_input['right_button'],
            states={'goal_relevant': False}, evidence='应用输入结构的相邻独立控件；不具备目标权限'))


def _append_audited_input_controls(elements: list[dict[str, Any]], *, controls: _AuditedInputControls,
    active_field_id: str, predecessor_field_id: str, active_field_label: str,
    input_element_id: str='local_audited_input_1') -> None:
    step = controls.step
    next_field = controls.next_field
    if next_field is not None:
        elements.append(_next_field_key_element(next_field, source_field_id=predecessor_field_id,
            target_field_id=active_field_id, target_field_label=active_field_label,
            input_element_id=input_element_id))
    if step is not None:
        common = {'prior_input_value': step.current_text, 'input_element_id': input_element_id}
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
            'input_element_id': input_element_id}, evidence='键盘区域内方向明确的独立输入模式切换键'))


def _optional_right_button(value: Any, *, input_bounds: NormalizedBounds) -> dict[str, Any] | None:
    if (not isinstance(value, dict) or _contains_action_like_wire_key(value)
        or (not _valid_1000_bounds(value.get('bounds')))):
        return None
    confidence = _diagnostic_confidence(value.get("confidence"))
    label = str(value.get("label") or "").strip()
    bounds = tuple(float(part) for part in value["bounds"])
    width = input_bounds[2] - input_bounds[0]
    if (not label or (not _bounds_inside(bounds, input_bounds,
        tolerance=20)) or (bounds[0] <= input_bounds[0] + 0.55 * width) or (_vertical_overlap_ratio(bounds,
        input_bounds) < 0.8)):
        return None
    return {'label': label, 'bounds': [round(part) for part in bounds], 'confidence': confidence}


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

    qwerty_geometry = None
    raw_anchors = keyboard.get("qwerty_anchors")
    if raw_anchors is not None and keyboard_visible and keyboard_layout == 'qwerty' and keyboard_bounds is not None:
        try:
            qwerty_geometry = _validated_qwerty_keyboard_geometry(snapped_anchors or raw_anchors,
                keyboard_bounds=keyboard_bounds, locally_snapped=snapped_anchors is not None)
        except UISceneError:
            qwerty_geometry = None
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
    mode_switch = keyboard_controls['mode']
    switch_is_goal = bool(switch_is_goal or (needs_mode_switch and mode_switch is not None
        and mode_switch['target_mode'] == required_mode))

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
    return _AuditedInputControls(step=step, candidate=exact_ime_candidate, preedit=exact_ime_preedit,
        qwerty=qwerty_geometry, backspace=backspace, clearable_preedit=clearable_preedit, next_field=next_field_key,
        required_mode=required_mode, switch_is_goal=switch_is_goal, mode=mode_switch, literal=exact_literal,
        enter=exact_enter, layout=exact_layout, case=exact_case)


def _apply_input_structure_audit(scene: UIScene, raw: str, *, fingerprint: str, goal_context: dict[str, Any],
    qwerty_row_snapper: Callable[[list[Image.Image] | tuple[Image.Image, ...], dict[str, Any]],
    dict[str, list[int]] | None] | None=None, qwerty_row_frames: list[Image.Image] | tuple[Image.Image, ...] | None=None,
    single_step_input_surface: dict[str, Any] | None=None) -> UIScene:
    try:
        goal = _goal_view(goal_context)
        payload = _normalize_input_structure_payload(_extract_json_object(raw))
        application_inputs = payload.get("application_inputs")
        ime_preedit_regions = payload.get("ime_preedit_regions")
        keyboard = payload.get("keyboard")
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
        trusted_preedits = _trusted_ime_preedits(ime_preedit_regions, keyboard_visible=keyboard_visible,
            keyboard_layout=keyboard_layout, keyboard_bounds=application_keyboard_bounds,
            qwerty_anchors=locally_snapped_qwerty_anchors or raw_audited_qwerty_anchors)
        active_field_id, active_field_label, active_multiline = goal.field
        active_transaction_text = goal.transaction_text
        matches = _collect_audited_input_matches(application_inputs)

        switch_is_goal = goal.mode_switch_requested
        active_clear_goal = goal.clear_requested
        trusted_input = _unique_audited_input(matches, label=active_field_label)
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
        if (trusted_input is not None and active_clear_goal and isinstance(trusted_input.get('text'),
            str) and trusted_input['text'] and isinstance(trusted_input.get('caret_line_index'), int)):
            extra_clear_units = max(0, trusted_input['caret_line_index'] - trusted_input['text'].count('\n'))
            if extra_clear_units > 0:
                trusted_input = dict(trusted_input)
                trusted_input["clear_extra_delete_units"] = extra_clear_units
        controls = _plan_audited_input_controls(trusted_input=trusted_input, predecessor_input=predecessor_input,
            trusted_preedits=trusted_preedits, goal=goal, active_clear_goal=active_clear_goal,
            active_field_id=active_field_id, active_multiline=active_multiline, keyboard=keyboard,
            keyboard_visible=keyboard_visible, keyboard_layout=keyboard_layout, keyboard_input_mode=keyboard_input_mode,
            keyboard_case_mode=keyboard_case_mode, keyboard_bounds=keyboard_bounds,
            snapped_anchors=locally_snapped_qwerty_anchors, switch_is_goal=switch_is_goal)
        rendered_input = trusted_input or (predecessor_input if controls.next_field is not None else None)
        input_element_id = _projected_input_element_id(single_step_input_surface)

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
            input_context = bool(element.get('role') == 'input'
                or element.get('meaning') == 'application_text_input')
            if input_context:
                element_states = dict(element.get("states") or {})
                # Scene keeps response-local identity and passive page context, while input_structure remains the
                # sole authority for committed text, focus and executable input state.
                element["states"] = {key: part for key, part in element_states.items()
                    if key in {'goal_relevant', 'fully_visible', 'visible', 'enabled'}}
                if (str(element.get('element_id') or '').strip() == input_element_id
                    and (rendered_input is not None or focus_only_input is not None)):
                    continue
            elements.append(element)
        if (trusted_input is None and controls.next_field is None and (controls.mode is None
            or not controls.switch_is_goal)):
            if focus_only_input is not None:
                elements.append(focus_only_input)
            value["elements"] = elements
            value["summary"] = (
                "当前input_structure未建立可执行输入字段；" "仅保留唯一粗编辑面用于聚焦，不授权文字、清空或发送。"
                if focus_only_input is not None
                else "当前input_structure未建立可执行输入字段；scene输入摘要不提供正文权威。"
            )
            return UIScene.from_dict(value, coordinate_scale=1.0, stable_override=True,
                fingerprint_override=fingerprint)
        if rendered_input is not None:
            _append_audited_input_element(elements, rendered_input,
                field_id=predecessor_field_id if controls.next_field is not None else active_field_id,
                field_label=predecessor_field_label if controls.next_field is not None else active_field_label,
                multiline=active_multiline, active_clear_goal=active_clear_goal, keyboard=audited_keyboard,
                controls=controls, input_element_id=input_element_id)
        _append_audited_input_controls(elements, controls=controls, active_field_id=active_field_id,
            predecessor_field_id=predecessor_field_id, active_field_label=active_field_label,
            input_element_id=input_element_id)
        value["elements"] = elements
        value["summary"] = "输入正文逐字来自当前input_structure；同响应scene输入元素身份保持不变。"
        return UIScene.from_dict(value, coordinate_scale=1.0, stable_override=True, fingerprint_override=fingerprint)
    except (UISceneError, ValueError, TypeError) as exc:
        raise VisionAgentError(f"输入结构只读审计结果不符合协议：{exc}") from exc


def _diagnostic_confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    confidence = float(value)
    return confidence if 0.0 <= confidence <= 1.0 else 0.0


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
        dict) or _contains_action_like_wire_key(value) or not (fields - {'confidence',
        'fully_visible'}).issubset(value) or (value.get('fully_visible') is False) or (value.get('key_action') not
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
    confidence = _diagnostic_confidence(value.get("confidence"))
    raw_bounds = value.get("bounds")
    if (not isinstance(raw_bounds, (list, tuple)) or len(raw_bounds) != 4 or any((isinstance(part,
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
        del field_name
        required = fields - {'confidence', 'fully_visible'}
        if (keyboard_bounds is None or not isinstance(value, dict) or _contains_action_like_wire_key(value)
            or not required.issubset(value) or (require_visible and value.get('fully_visible') is False)
            or not _valid_1000_bounds(value.get('bounds'))):
            return None
        confidence = _diagnostic_confidence(value.get('confidence'))
        label = str(value.get('label') or '').strip()
        bounds = tuple(float(part) for part in value['bounds'])
        if not label or not _bounds_inside(bounds, keyboard_bounds, tolerance=tolerance):
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
        del field_name, item_error
        if not isinstance(value, list) or (value and keyboard_bounds is None):
            return []
        required = fields - {'confidence', 'fully_visible'}
        return [item for item in value[:maximum] if isinstance(item, dict) and required.issubset(item)
            and not _contains_action_like_wire_key(item)]

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
        if (not isinstance(key_value, str) or len(key_value) != 1 or not isinstance(label, str)
            or key_kind not in {'character', 'space'} or not _valid_1000_bounds(item.get('bounds'))
            or (item.get('fully_visible') is False)):
            continue
        if (key_kind == 'character' and (key_value == ' ' or label != key_value)
            or key_kind == 'space' and (key_value != ' ' or label.strip().casefold() not in {'', 'space', '空格'})):
            # Revoke a mismatched optional glyph without vetoing independent input or layout evidence.
            continue
        bounds = tuple(float(part) for part in item['bounds'])
        confidence = _diagnostic_confidence(item.get('confidence'))
        if (keyboard_bounds is None
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
