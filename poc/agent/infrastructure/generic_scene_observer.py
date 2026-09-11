"""Local visual-observation adapter for the generic Agent runtime."""

from __future__ import annotations

from agent.domain.validation import NormalizedBounds, reject_if
import json
from pathlib import Path
from uuid import uuid4
from functools import lru_cache
from importlib.resources import files
import math
import threading
import time
import os
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from typing import Any

from PIL import Image
from agent.infrastructure.atomic_files import atomic_replace_bytes, json_bytes
from agent.infrastructure.model_failure_diagnostics import redact_model_failure_response
from agent.infrastructure.observation_images import (
    consensus_top_edge_obstructions,
    local_frame_fingerprint,
    measure_frame_sharpness,
    measure_local_stability,
)
from agent.domain.canonical_action_kinds import CANONICAL_ACTION_KINDS
from agent.domain.canonical_action_protocol import (
    CanonicalActionProtocolError,
    MODEL_STEP_DIRECT_POINT_ACTIONS,
    normalize_model_step_decision,
)
from agent.infrastructure.qwen_runtime_errors import classify_qwen_error
from agent.domain.ui_scene import (
    UI_SCENE_PROTOCOL_VERSION,
    UIElement,
    UIScene,
    UISceneError,
    camera_alignment_evidence_is_safe,
)
from agent.infrastructure.dashscope_vision_provider import _extract_json_object, _image_data_url, _image_request_size
from agent.domain.vision_model import VisionAgentError, public_model_identity

import agent.domain.generic_goal as generic_goal_domain

SINGLE_STEP_SCENE_OBSERVER_VERSION = "2026-09-02-single-step-scene-action-finish-v9"
SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION = "2026-09-07-single-step-current-frame-v22"
INPUT_STRUCTURE_AUDIT_VERSION = "2026-09-06-input-structure-field-preedit-v17"
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
def _positive_env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and value > 0 else default


def _positive_env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


# One request per fresh scene remains mandatory. These values only bound the
# cloud wait and transient retry cost; they never skip the post-action scene.
OBSERVATION_TIMEOUT_SECONDS = _positive_env_float("VISION_OBSERVATION_TIMEOUT_SECONDS", 90.0)
OBSERVATION_MAX_ATTEMPTS = _positive_env_int("VISION_OBSERVATION_MAX_ATTEMPTS", 1)

_ACTION_LIKE_WIRE_KEYS = frozenset({'action', 'actions', 'plan', 'plans', 'step', 'steps', 'tap', 'swipe',
    'command', 'shell', 'coordinates', 'next_action', 'execution_plan'})

_INPUT_AUDIT_FIELDS = frozenset({'element_id', 'structure_id', 'bounds', 'fully_visible', 'text', 'preedit_text', 'placeholder',
    'visible_editable_cues', 'caret_line_index', 'focused', 'confidence', 'right_button'})




def _normalize_input_structure_payload(value: Any) -> dict[str, Any]:
    if value is None:
        value = {}
    reject_if(not isinstance(value, Mapping), UISceneError("input_structure必须是对象或null。"))
    allowed = {'protocol_version', 'application_inputs'}
    reject_if(_contains_action_like_extra(value, allowed), UISceneError("input_structure包含动作或计划字段。"))
    reject_if(value.get('protocol_version', INPUT_STRUCTURE_AUDIT_VERSION) != INPUT_STRUCTURE_AUDIT_VERSION,
        UISceneError("输入结构审计协议版本不匹配。"))
    inputs = value.get('application_inputs', [])
    return {'protocol_version': INPUT_STRUCTURE_AUDIT_VERSION,
        'application_inputs': list(inputs) if isinstance(inputs, list) else []}


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


STAGE_LABELS = {'idle': '空闲', 'checking_stability': '检查当前画面', 'waiting_single_step_observation': '等待千问单步完整观察',
    'parsing_single_step_observation': '解析单步完整观察', 'completed': '观察完成', 'failed': '观察安全停止'}


class SingleStepGenericSceneObserver():
    """Use one Qwen envelope per fresh scene as the only scene/input observation authority."""

    def __init__(self, provider: Any) -> None:
        self.provider = provider
        self.last_raw_response = ""
        self.last_diagnostics: dict[str, Any] = {}
        self._stage_lock = threading.RLock()
        self._current_stage = "idle"
        self._last_stage = "idle"
        self.supports_runtime_action_contract = True
        self.supports_response_evidence = True
        self.last_response_evidence_path: str | None = None

    def _set_stage(self, stage: str) -> None:
        with self._stage_lock:
            self._current_stage = stage
            if stage != 'idle':
                self._last_stage = stage

    def _provider_chat(self, messages: list[dict[str, Any]], *, max_tokens: int | None,
        response_format: dict[str, Any] | None=None) -> str:
        return self.provider._chat(messages, max_tokens=max_tokens, timeout=OBSERVATION_TIMEOUT_SECONDS,
            max_attempts=OBSERVATION_MAX_ATTEMPTS,
            response_format=response_format or {'type': 'json_object'})

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
            'observation_max_attempts': OBSERVATION_MAX_ATTEMPTS,
            'last_scene_enum_values': dict(self.last_diagnostics.get('scene_enum_values') or {}),
            'last_input_structure_shape': dict(self.last_diagnostics.get('input_structure_shape') or {})})
        return value

    def observe(self, *, frames: list[Image.Image], goal_context: dict[str, Any] | None=None,
        device_id: str | None=None, available_action_kinds: Iterable[str] | None=None) -> UIScene:
        scene, _ = self.observe_with_decision(frames=frames, goal_context=goal_context, device_id=device_id,
            available_action_kinds=available_action_kinds)
        return scene

    def observe_with_decision(self, *, frames: list[Image.Image], goal_context: dict[str, Any] | None=None,
        device_id: str | None=None, available_action_kinds: Iterable[str] | None=None, response_evidence_dir: Path | None=None,
        response_evidence_prefix: str='observation') -> tuple[UIScene, dict[str, Any]]:
        self.last_raw_response = ""
        model_identity = public_model_identity(self.provider.status())
        self.last_diagnostics = {"vision_model": model_identity,
            "foreground_identity_source": "qwen_visual"}
        self._set_stage("checking_stability")
        started = time.perf_counter()
        model_calls = 0
        fingerprint = ""
        self.last_response_evidence_path = None
        selected_frame_index = 0
        stable_tail_start = 0
        input_structure_required = False
        try:
            reject_if(len(frames) < 4, VisionAgentError("通用页面观察至少需要4帧。"))
            stability = measure_local_stability(frames, allow_leading_outlier=True)

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
            response_format = _single_step_response_format(context,
                input_structure_required=input_structure_required, request_height=request_image_size[1],
                available_action_kinds=runtime_actions)

            prompt = _single_step_observation_prompt(context, include_input_structure=input_structure_required,
                image_count=len(model_frames), request_image_size=request_image_size,
                available_action_kinds=runtime_actions)
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for (index, item) in enumerate(model_frames, start=1):
                content.extend(({'type': 'text', 'text': f'IMAGE {index} - CURRENT PHONE SURFACE'},
                    {'type': 'image_url', 'image_url': {'url': _image_data_url(item.convert('RGB'))}}))
            self._set_stage("waiting_single_step_observation")
            scope_factory = getattr(self.provider, "call_scope", None)
            response_id = uuid4().hex
            request_evidence_path = (Path(response_evidence_dir) / f"{response_id}_model_request.json"
                if response_evidence_dir is not None else None)
            scope = scope_factory(stage='single_step_observation',
                fingerprint=fingerprint, request_evidence_path=request_evidence_path) if callable(scope_factory) else nullcontext()
            call_started = time.perf_counter()
            model_calls = 1
            with scope:
                raw = self._provider_chat([_json_only_system_message(), {'role': 'user', 'content': content}],
                    max_tokens=None, response_format=response_format)
            call_elapsed = round(time.perf_counter() - call_started, 3)
            self.last_raw_response = raw
            if response_evidence_dir is not None:
                # Save before any parse/projection/binding, including successful observations.
                evidence_dir = Path(response_evidence_dir)
                evidence_dir.mkdir(parents=True, exist_ok=True)
                # Keep step provenance in the record, not in an unbounded Windows path.
                evidence_path = evidence_dir / f"{response_id}_model_response.json"
                self.last_response_evidence_path = str(atomic_replace_bytes(evidence_path, json_bytes({
                    'artifact_version': '2026-09-04-preparse-model-response-v1',
                    'response_id': response_id, 'device_id': device_id, 'fingerprint': fingerprint,
                    'request_evidence_path': str(request_evidence_path) if request_evidence_path is not None
                        and request_evidence_path.is_file() else None,
                    'response_evidence_prefix': redact_model_failure_response(response_evidence_prefix),
                    'observer_protocol': SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    'goal_context': json.loads(redact_model_failure_response(json.dumps(context, ensure_ascii=False))),
                    'raw_response_length': len(raw), 'redacted_response_truncated': False,
                    'redacted_raw_response': redact_model_failure_response(raw),
                })))
            self._set_stage("parsing_single_step_observation")
            envelope = _parse_single_step_observation_envelope(raw, input_structure_required=input_structure_required,
                request_image_size=request_image_size)
            model_decision = dict(envelope["decision"])
            scene_payload = dict(envelope["scene"])
            model_foreground_app_id = str(scene_payload.get("foreground_app_id")
                or scene_payload.get("app_id") or "unknown").strip()
            obstructions = consensus_top_edge_obstructions(frames[stable_tail_start:])
            referenced_element_ids = _decision_element_ids(model_decision)
            scene = _parse_scene(json.dumps(scene_payload, ensure_ascii=False, separators=(',', ':')),
                fingerprint=fingerprint, camera_layout_orientation=_camera_layout_orientation(frame),
                strict_element_ids=referenced_element_ids)

            if input_structure_required:
                input_payload = envelope["input_structure"]
                assert isinstance(input_payload, dict)
                scene = _apply_input_structure_audit(scene, json.dumps(input_payload,
                    ensure_ascii=False, separators=(',', ':')), fingerprint=fingerprint, goal_context=context)


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
                'stable_tail_start_index': stable_tail_start, 'local_stability': stability.to_dict(), 'temporal_change_is_error': False,
                'frame_sharpness_scores': [round(value, 3) for value in sharpness_scores],
                'frame_size': list(frame.size), 'request_image_size': list(request_image_size),
                'coordinate_normalization': envelope.get('coordinate_normalization'), 'fingerprint': fingerprint,
                'element_count': len(scene.elements), 'decision_status': model_decision['status'],
                'foreground_identity_source': 'qwen_visual',
                'model_foreground_app_id': model_foreground_app_id,
                'available_action_kinds': list(runtime_actions),
                'local_visual_obstructions': [item.to_dict() for item in obstructions],
                'model_call_elapsed_seconds': [call_elapsed],
                'model_call_token_budgets': [None],
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
                'foreground_identity_source': 'qwen_visual',
                'error_type': classify_qwen_error(exc, raw_response=self.last_raw_response),
                'safe_stop_reason': '单次模型输出未建立完整可信观察；没有发起第二次Qwen请求，控制器与机械臂均未执行。',
                'raw_response_length': len(self.last_raw_response),
                'raw_response_excerpt': self.last_raw_response[:1000],
                'elapsed_seconds': round(time.perf_counter() - started, 3)}
            raise
        finally:
            self._set_stage("idle")


def _json_only_system_message() -> dict[str, str]:
    return {'role': 'system', 'content': '你是通用手机视觉操作Agent。根据用户整任务、本会话实际执行历史和本轮截图报告画面事实，并按协议选择一个canonical动作或整任务finish。只输出一个语法完整且符合响应schema的JSON对象；禁止Markdown、JSON之外的解释或思考过程、代码围栏、JSON字符串套壳或对象前后的任何文字。'}


LOCAL_TEXT_CLEAR_OBSERVATION_RULE = (
    "若非空输入框内部或紧邻右侧清楚可见独立的圆形×/清空图标，必须另建role=button或icon元素，"
    "meaning写clear_local_text，states写local_text_clear:true，label必须逐字写图标本身的×/✕/✖/x；"
    "若看不清真实叉号图形或只能自由描述为叉号，就不得标记local_text_clear。只框该图标自身，不能与输入框合并，"
    "也绝不能把键盘退格键/删除键标成local_text_clear。页面右侧的文字‘取消’/cancel是取消编辑或"
    "退出控件，不是本地清空图标；必须meaning=cancel且goal_relevant:false，绝不能标成clear_local_text。"
)

def _single_step_observation_prompt(context: dict[str, Any], *, include_input_structure: bool,
    image_count: int, request_image_size: tuple[int, int],
    available_action_kinds: tuple[str, ...]) -> str:
    request_width, request_height = request_image_size
    current_goal = _goal_view(context).observation_context
    scene_contract = _compact_prompt({}, wire_height=1000,
        input_structure_is_value_authority=include_input_structure)
    if include_input_structure:
        input_contract = _input_structure_audit_prompt({}, wire_height=1000)
        input_rule = ("input_structure报告下面INPUT CONTRACT中当前可见的最小事实；空输入只需合法bounds和"
            "text=\"\"，无关可选字段、整套键盘几何和另一个scene输入框均不必补齐。")
    else:
        input_contract = "本次显式只读观察不要求输入结构。"
        input_rule = "input_structure必须为null，不得额外枚举键盘或输入结构。"
    temporal_rule = (f"共有{image_count}张本轮观察的连续手机画面。只把它们作为本轮当前状态的证据；"
        "闪烁光标可从任一帧读取，不把互不相容的瞬态拼成一个状态。" if image_count > 1 else "只有一张本轮当前手机画面。")
    temporal_rule += "视频、动画或跨帧像素变化本身不是错误，不要求画面静止；按本轮可见事实判断。"
    return _render_prompt("single_step_observation.txt", SCENE_CONTRACT=scene_contract,
        INPUT_CONTRACT=input_contract, TEMPORAL_RULE=temporal_rule,
        INPUT_RULE=input_rule, REQUEST_WIDTH=str(request_width), REQUEST_HEIGHT=str(request_height),
        OBSERVATION_PROTOCOL=SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION, SCENE_PROTOCOL=UI_SCENE_PROTOCOL_VERSION,
        AVAILABLE_ACTIONS_JSON=json.dumps(list(available_action_kinds), ensure_ascii=False, separators=(',', ':')),
        WHOLE_TASK_JSON=json.dumps(current_goal, ensure_ascii=False, separators=(',', ':')))


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


def _single_step_response_format(context: dict[str, Any], *, input_structure_required: bool,
    request_height: int, available_action_kinds: tuple[str, ...]) -> dict[str, Any]:
    """Constrain impossible pending-transition finishes before Qwen generates them."""

    input_schema: dict[str, Any]
    if input_structure_required:
        input_schema = {
            'type': 'object',
            'properties': {
                'protocol_version': {'type': 'string', 'enum': [INPUT_STRUCTURE_AUDIT_VERSION]},
                'application_inputs': {'type': 'array', 'description': '当前选中输入框的事实，无需元素编号。',
                    'items': {'type': 'object', 'properties': {
                        'bounds': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 4, 'maxItems': 4},
                        'multiline': {'type': ['boolean', 'null']}, 'text': {'type': 'string'}, 'preedit_text': {'type': 'string'}, 'focused': {'type': ['boolean', 'null']}},
                    'required': ['bounds', 'focused', 'preedit_text'], 'additionalProperties': True}},
            },
            'required': ['protocol_version', 'application_inputs'],
            'additionalProperties': False,
        }
    else:
        input_schema = {'type': 'null'}
    scene_schema = {
        'type': 'object',
        'properties': {
            'protocol_version': {'type': 'string', 'enum': [UI_SCENE_PROTOCOL_VERSION]},
            'foreground_app_id': {'type': 'string'},
            'screen_id': {'type': 'string'},
            'summary': {'type': 'string'},
            'system_ui': {'type': 'object', 'additionalProperties': True},
            'camera_alignment': {'type': 'object', 'additionalProperties': True},
            'elements': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': True},
                },
            'overlays': {'type': 'array', 'items': {'type': 'string'}},
            'stable': {'type': 'boolean'},
            'confidence': {'type': 'number'},
            'fingerprint': {'type': 'string'},
        },
        'required': ['protocol_version', 'foreground_app_id', 'screen_id', 'summary', 'system_ui',
            'camera_alignment', 'elements', 'overlays', 'stable', 'confidence', 'fingerprint'],
        'additionalProperties': False,
    }
    direct_actions = sorted(set(available_action_kinds) & set(MODEL_STEP_DIRECT_POINT_ACTIONS))
    other_actions = sorted(set(available_action_kinds) - set(MODEL_STEP_DIRECT_POINT_ACTIONS))
    common_diagnostics = {
        'confidence': {'type': 'number'},
        'reason': {'type': 'string'},
    }
    decision_properties: dict[str, Any] = {
        'status': {'type': 'string', 'enum': ['action', 'finish']},
        'action': {'type': ['string', 'null'], 'enum': [*available_action_kinds, None]},
    }
    if direct_actions:
        decision_properties.update({
            'target': {
                'type': 'object',
                'properties': {
                    'role': {'type': 'string'},
                    'meaning': {'type': 'string'},
                    'label': {'type': 'string'},
                    'evidence': {'type': 'array', 'items': {'type': 'string'}},
                },
                'required': ['role', 'meaning'],
                'additionalProperties': False,
                'description': '点按动作的唯一目标身份，不含bounds；其他动作和finish为null。',
            },
            'tap_point': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2},
        })
    decision_properties.update({
        "text": {"type": ["string", "null"]},
        "app": {"type": ["string", "null"]},
        "previous_action_outcome": {"type": ["string", "null"], "enum": ["matched", "unmatched", "uncertain", None]},
        "state_action_consistent": {"type": ["boolean", "null"],
            "description": "Qwen对当前明确状态与所选动作的一次一致性判断；矛盾时为false。"},
        "postcondition": {"type": ["object", "null"], "properties": {
            "status": {"type": "string", "enum": ["confirmed", "not_confirmed", "unknown", "not_applicable"]},
            "fact": {"type": "string"}}, "required": ["status", "fact"], "additionalProperties": False,
            "description": "动作后当前目标状态；外部效果必须明确报告，无法判断填unknown。"},
    })
    decision_properties.update(common_diagnostics)
    if other_actions:
        decision_properties.update({
            'element_id': {'type': 'string'},
            'source_element_id': {'type': 'string'},
            'destination_element_id': {'type': 'string'},
            'direction': {'type': 'string', 'enum': ['up', 'down', 'left', 'right'],
                'description': '仅scroll使用；scroll只能填写direction，其他轨迹字段必须为null。'},
            'start': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2,
                'description': '仅swipe_element使用；scroll必须为null。'},
            'end': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2,
                'description': '仅swipe_element使用；scroll必须为null。'},
        })
    # Cross-field action semantics remain with the single canonical parser.
    # Do not rely on provider support for conditional JSON Schema composition.
    for name in ('target', 'tap_point', 'element_id', 'source_element_id',
            'destination_element_id', 'direction', 'start', 'end'):
        if name not in decision_properties:
            decision_properties[name] = {'type': 'null'}
        else:
            decision_properties[name]['type'] = [decision_properties[name]['type'], 'null']
            if 'enum' in decision_properties[name]:
                decision_properties[name]['enum'].append(None)
    scene_schema['properties']['elements']['description'] = (
        'tap_semantic/dismiss_overlay/double_tap/long_press时请留空；额外列表只作原始诊断，不参与执行；'
        '其他动作或finish才可报告带bounds的元素。')
    schema = {
        'type': 'object',
        'properties': {
            'protocol_version': {'type': 'string', 'enum': [SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION]},
            'coordinate_space': {
                'type': 'object',
                'properties': {
                    'kind': {'type': 'string', 'enum': ['axis_grid']},
                    'width': {'type': 'integer', 'enum': [1000]},
                    'height': {'type': 'integer', 'enum': [1000]},
                },
                'required': ['kind', 'width', 'height'],
                'additionalProperties': False,
            },
            'scene': scene_schema,
            'input_structure': input_schema,
            'decision': {'type': 'object', 'properties': decision_properties,
                'required': list(decision_properties), 'additionalProperties': False},
        },
        'required': ['protocol_version', 'coordinate_space', 'scene', 'input_structure', 'decision'],
        'additionalProperties': False,
    }
    return {'type': 'json_schema', 'json_schema': {
        'name': 'current_scene_observation',
        'strict': True,
        'schema': schema,
    }}


def _decision_element_ids(decision: Mapping[str, Any]) -> tuple[str, ...]:
    """Return every scene element that this same-envelope decision makes authoritative."""

    if decision.get('status') == 'action' and decision.get('action') in MODEL_STEP_DIRECT_POINT_ACTIONS:
        # Direct target identity is independent of model scene elements/evidence_refs.
        # Typed point targets are bound to the same-frame input audit downstream.
        return ()
    values = [str(decision.get(name) or '').strip() for name in (
        'element_id', 'source_element_id', 'destination_element_id') if decision.get(name)]
    return tuple(dict.fromkeys(values))


def _normalize_single_step_wire_coordinates(payload: dict[str, Any], *, request_image_size: tuple[int,
    int], decision: Mapping[str, Any]) -> dict[str, Any]:
    """Convert the sole request-bound axis grid to canonical 0..1000."""
    coordinate_space = payload.get("coordinate_space")
    request_width, request_height = request_image_size
    expected = {"kind": "axis_grid", "width": 1000, "height": 1000}
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
            and 0 <= left < right <= 1000 and 0 <= top < bottom <= 1000)
        if not valid_extent:
            reject_if(selected, UISceneError("已选目标的bounds超出声明的axis_grid。"))
            return None
        result = [round(left), round(top), round(right), round(bottom)]
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

    def normalized_point(value: Any, *, label: str) -> list[int]:
        valid_shape = bool(isinstance(value, (list, tuple)) and len(value) == 2
            and all(isinstance(part, (int, float)) and not isinstance(part, bool) for part in value))
        reject_if(not valid_shape, UISceneError(f"{label}格式无效。"))
        x, y = (float(part) for part in value)
        reject_if(not (math.isfinite(x) and math.isfinite(y)
            and 0 <= x <= 1000 and 0 <= y <= 1000),
            UISceneError(f"{label}超出声明的axis_grid。"))
        return [round(x), round(y)]

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

    if decision.get('action') in MODEL_STEP_DIRECT_POINT_ACTIONS:
        decision['tap_point'] = normalized_point(decision.get('tap_point'),
            label=f"{decision.get('action')}.tap_point")
    if decision.get('action') == 'scroll':
        # Qwen sometimes echoes a swipe-like start/end pair while also giving
        # a valid scroll direction.  The executor's scroll contract is the
        # direction; discard only those redundant fields, never reinterpret
        # the action as a different gesture.
        decision['start'] = None
        decision['end'] = None
    if decision.get('action') == 'swipe_element':
        decision['start'] = normalized_point(decision.get('start'), label='swipe_element.start')
        decision['end'] = normalized_point(decision.get('end'), label='swipe_element.end')

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
        direct_target = decision.get('target') if isinstance(decision.get('target'), Mapping) else {}
        selected_input = bool(decision.get('status') == 'action' and (
            decision.get('action') in {'input_verified_text', 'clear_verified_text', 'press_enter'}
            or direct_target.get('role') == 'input'))
        reject_if(selected_input and not normalized_inputs and invalid_inputs > 0,
            UISceneError("已选输入框的bounds无效。"))
        input_structure['application_inputs'] = normalized_inputs

    payload.pop("coordinate_space", None)
    return {'wire_kind': 'axis_grid', 'wire_extent': [1000, 1000], 'request_image_size': [request_width,
        request_height], 'canonical_extent': [1000, 1000], 'applied': True}


def _parse_single_step_observation_envelope(raw: str, *, input_structure_required: bool, request_image_size: tuple[int,
    int]) -> dict[str, Any]:
    """Parse one current-scene response without remote repair or resampling."""

    try:
        payload = _extract_json_object(raw, reject_duplicate_keys=True,
            unwrap_singleton_object_array=True)
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
        if decision['status'] == 'action' and decision['action'] in MODEL_STEP_DIRECT_POINT_ACTIONS:
            # Project once, before recursive field/geometry/input checks. All downstream
            # consumers receive this scene; last_raw_response retains the untouched wire.
            # Never use extra model elements to repair, compare or veto target/tap_point.
            payload['scene'] = {**payload['scene'], 'elements': []}
        coordinate_normalization = _normalize_single_step_wire_coordinates(payload,
            request_image_size=request_image_size, decision=decision)
        return {'scene': payload['scene'], 'input_structure': payload['input_structure'],
        'decision': decision, 'coordinate_normalization': coordinate_normalization}
    except (CanonicalActionProtocolError, UISceneError, ValueError, TypeError) as exc:
        raise VisionAgentError(f"单步完整观察结果不符合协议：{exc}") from exc


def _compact_prompt(context: dict[str, Any], *, wire_height: int=1000,
    input_structure_is_value_authority: bool=False) -> str:
    context = _goal_view(context).observation_context
    input_rule = (("本轮input_structure.application_inputs.text是应用输入正文空/非空事实的"
        "唯一视觉权威。scene中的role=input只是可选页面上下文，不必与input_structure重复出现或"
        "提供正文、占位符、光标、证据、置信度阈值或几何重合；不得用scene事实否决input_structure。"
        "同一字段的未提交预编辑只由input_structure.application_inputs.preedit_text报告，不推导键盘动作。"
        + LOCAL_TEXT_CLEAR_OBSERVATION_RULE) if input_structure_is_value_authority else
        "输入框只作页面上下文，不规划键盘文字动作。" + LOCAL_TEXT_CLEAR_OBSERVATION_RULE)
    foreground_identity_rule = ("桌面写 launcher；系统最近任务页面必须写foreground_app_id=system、"
        "screen_id=system_recent_tasks。看到应用卡片叠放/缩略图、底部圆形X清理按钮或系统最近任务导航栏时，"
        "优先按最近任务页面识别，即使卡片预览内容是微信聊天或其它App，也不能把卡片内容当成当前前台页面。"
        "不确定写 unknown。不得把目标App当成当前App，也不得把"
        "current_foreground、current_app、foreground_app、target_app 或 active_app 等引用占位符"
        "写成foreground_app_id；该字段只能来自当前画面的视觉身份。")
    return _render_prompt("compact_scene.txt", CONTEXT=json.dumps(context, ensure_ascii=False, separators=(',', ':')),
        INPUT_RULE=input_rule, WIRE_HEIGHT=str(wire_height),
        SCENE_PROTOCOL=UI_SCENE_PROTOCOL_VERSION, FOREGROUND_IDENTITY_RULE=foreground_identity_rule)




def _input_structure_audit_prompt(context: dict[str, Any], *, wire_height: int=1000) -> str:
    return _render_prompt("input_structure_audit.txt",
        CONTEXT=json.dumps(_goal_view(context).observation_context, ensure_ascii=False, separators=(',', ':')),
        WIRE_HEIGHT=str(wire_height), AUDIT_VERSION=INPUT_STRUCTURE_AUDIT_VERSION)


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
        'label': str(raw_element.get('label') or '').strip(),
        'states': {},
        'evidence': [item.strip() for item in raw_element.get('evidence', [])
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










def _audited_element(suffix: str, meaning: str, source: Mapping[str, Any], *, states: Mapping[str, Any],
    evidence: str | Iterable[str], role: str='button', label: str | None=None, bounds_key: str='bounds') -> dict[str,
    Any]:
    state = {'fully_visible': True, **states, 'primary_input_geometry_verified': True,
        'geometry_audit_source': 'input_structure_audit'}
    return {'element_id': f'local_audited_{suffix}_1', 'role': role, 'meaning': meaning, 'label': source.get('label',
        '') if label is None else label, 'bounds': [part / 1000.0 for part in source[bounds_key]],
        'confidence': source['confidence'], 'states': state, 'evidence': [evidence] if isinstance(evidence,
        str) else list(evidence)}






def _audit_strings(value: Any, *, name: str, allow_empty: bool) -> tuple[str, ...]:
    reject_if(
        not isinstance(value, list) or any(not isinstance(item, str)
            or (not allow_empty and not item.strip()) for item in value),
        UISceneError(f"{name} 不符合输入结构协议。"),
    )
    return tuple(dict.fromkeys(item.strip() for item in value if item.strip()))


def _optional_audit_strings(value: Any, *, name: str, allow_empty: bool) -> tuple[str, ...]:
    """Project optional diagnostics without letting their shape veto valid input geometry."""

    try:
        return _audit_strings(value, name=name, allow_empty=allow_empty)
    except UISceneError:
        return ()






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
        if not isinstance(text, str):
            continue
        cues = list(_optional_audit_strings(item.get('visible_editable_cues', []),
            name='visible_editable_cues', allow_empty=True))
        labels = _optional_audit_strings(item.get('field_labels', []), name='field_labels',
            allow_empty=False)
        caret = item.get("caret_line_index")
        if caret is not None and (isinstance(caret, bool) or not isinstance(caret, int) or not 0 <= caret <= 30):
            caret = None
        placeholder = item.get("placeholder")
        placeholder = placeholder.strip() if isinstance(placeholder, str) else ""
        bounds = tuple(float(part) for part in item["bounds"])
        button = _optional_right_button(item.get("right_button"), input_bounds=bounds)
        input_bounds = [round(part) for part in bounds]
        if button is not None:
            input_bounds[2] = button["bounds"][0]
        if input_bounds[2] <= input_bounds[0]:
            continue
        focus = item.get('focused')
        focus = focus if type(focus) is bool else None
        match = {'text': text, 'preedit_text': item.get('preedit_text', ''), 'placeholder': placeholder, 'focused': focus,
            'visible_editable_cues': cues, 'caret_line_index': caret,
            'field_labels': labels, 'input_bounds': input_bounds, 'right_button': button,
            'confidence': 1.0}
        reject_if(not isinstance(match['preedit_text'], str), UISceneError('当前字段preedit_text必须是字符串。'))
        matches.append(match)
    return matches


def _unique_audited_input(matches: Iterable[dict[str, Any]], *, text: str | None=None) -> dict[str,
    Any] | None:
    selected = [item for item in matches if text is None or item['text'] == text]
    return selected[0] if len(selected) == 1 else None


def _append_audited_input_element(elements: list[dict[str, Any]], audited_input: dict[str, Any], *,
    field_id: str, field_label: str, multiline: bool) -> None:
    states = {'goal_relevant': True, 'fully_visible': True, 'value': audited_input['text'],
        'ime_preedit_text': audited_input['preedit_text'], 'input_multiline': multiline}
    if field_id:
        states['input_field_id'] = field_id
    if field_label:
        states['input_field_label'] = field_label
    if type(audited_input.get('focused')) is bool:
        states['focused'] = audited_input['focused']
    if audited_input['placeholder']:
        states['placeholder'] = audited_input['placeholder']
    evidence = list(dict.fromkeys((*audited_input['field_labels'], *audited_input['visible_editable_cues'])))
    element = _audited_element('input', 'application_text_input', audited_input, role='input',
        label=audited_input['text'] or audited_input['placeholder'], bounds_key='input_bounds', states=states,
        evidence=evidence)
    element['element_id'] = 'local_audited_input_1'
    elements.append(element)
    if audited_input['right_button'] is not None:
        elements.append(_audited_element('adjacent_button', 'adjacent_input_utility', audited_input['right_button'],
            states={'goal_relevant': False}, evidence='应用输入结构的相邻独立控件；不具备目标权限'))




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




def _apply_input_structure_audit(scene: UIScene, raw: str, *, fingerprint: str,
    goal_context: dict[str, Any]) -> UIScene:
    try:
        goal = _goal_view(goal_context)
        payload = _normalize_input_structure_payload(_extract_json_object(raw))
        trusted_input = _unique_audited_input(_collect_audited_input_matches(payload['application_inputs']))
        value = scene.to_dict()
        elements = [dict(item) for item in value.get('elements') or [] if isinstance(item, dict)
            and item.get('role') != 'input' and item.get('meaning') != 'application_text_input']
        if trusted_input is not None:
            field_id, label, multiline = goal.field
            inputs = payload["application_inputs"]
            multiline = bool(len(inputs) == 1 and isinstance(inputs[0], dict) and inputs[0].get("multiline") is True)
            _append_audited_input_element(elements, trusted_input, field_id=field_id,
                field_label=label, multiline=multiline)
        value['elements'] = elements
        return UIScene.from_dict(value, coordinate_scale=1.0, stable_override=True, fingerprint_override=fingerprint)
    except (UISceneError, ValueError, TypeError) as exc:
        raise VisionAgentError(f"输入结构只读审计结果不符合协议：{exc}") from exc


def _diagnostic_confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    confidence = float(value)
    return confidence if 0.0 <= confidence <= 1.0 else 0.0






















def _bounds_inside(inner: NormalizedBounds, outer: NormalizedBounds, *,
    tolerance: float) -> bool:
    return inner[0] >= outer[0] - tolerance and inner[1] >= outer[1] - tolerance and (inner[2] <= outer[2] +
        tolerance) and (inner[3] <= outer[3] + tolerance)


def _vertical_overlap_ratio(first: NormalizedBounds, second: NormalizedBounds) -> float:
    overlap = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    smaller = min(first[3] - first[1], second[3] - second[1])
    return overlap / smaller if smaller > 0 else 0.0



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

