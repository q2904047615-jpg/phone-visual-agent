"""Pure domain service resolving and verifying canonical UI actions."""

from __future__ import annotations

from .validation import DataclassWire, NormalizedBounds, NormalizedPoint, bounds_overlap, dataclass_wire, reject_if
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import math
from typing import Any

from agent.domain.semantic_action import SemanticAction
from agent.domain.text_input_utils import editable_character_count
from agent.domain.verified_text_transaction import (
    VerifiedTextTransactionError,
    is_direct_latin_segment,
    plan_from_input_states,
)
from agent.domain.input_value_lineage import (
    input_app_identity_compatible,
    input_app_identity_is_concrete_package,
    input_screen_identity_compatible,
    input_screen_identity_family,
)
from agent.domain.ui_scene import (
    MIN_TARGET_CONFIDENCE,
    UIElement,
    UIScene,
    UISceneError,
    compact_drag_source_container_error,
    scene_matches_app_identity,
    scene_surface_kind,
)


UNIVERSAL_CONTROLLER_PROTOCOL_VERSION = "2026-08-26-universal-action-v17"

LOCAL_POINT_GROUNDING_SOURCE = "2026-08-25-stable-local-ocr-label-v1"
MAX_LOCAL_GROUNDING_BOX_GAP = 0.06

REVEAL_SYSTEM_NAVIGATION_EFFECT = {'system_ui': {'navigation_bar_visible': True}}

GESTURE_EDGE_MARGIN = 0.02
TARGETED_SWIPE_EDGE_MARGIN = 0.08
MIN_DRAG_DISTANCE = 0.08
MAX_DRAG_DISTANCE = 0.90
DRAG_DURATION_SECONDS = 0.8
MIN_DRAG_RESULT_DISPLACEMENT = 0.04
CONTROLLER_INPUT_PREEDIT_PENDING = "input_transition=preedit_pending"


class UniversalActionError(RuntimeError):
    pass


@dataclass(frozen=True)
class LocalPointGrounding(DataclassWire):
    """Stable local evidence refining one already-selected target point."""

    source: str
    scene_fingerprint: str
    element_id: str
    label: str
    model_bounds: NormalizedBounds
    proposed_point: NormalizedPoint
    grounded_bounds: NormalizedBounds
    grounded_point: NormalizedPoint
    matched_frames: int
    inspected_frames: int

    @staticmethod
    def _normalized_numbers(value: tuple[float, ...], length: int, label: str) -> tuple[float, ...]:
        reject_if(
            not isinstance(value, tuple) or len(value) != length or any((isinstance(part, bool) or not isinstance(part,
            (int, float)) or (not math.isfinite(float(part))) for part in value)),
            UniversalActionError(f"{label}格式无效。"),
        )
        numbers = tuple(float(part) for part in value)
        reject_if(not all((0.0 <= part <= 1.0 for part in numbers)), UniversalActionError(f"{label}超出归一化画面。"))
        reject_if(length == 4 and (not (numbers[0] < numbers[2] and numbers[1] < numbers[3])), UniversalActionError(f"{label}超出归一化画面。"))
        return numbers

    def validate_for(self, scene: UIScene, element: UIElement) -> None:
        reject_if(self.source != LOCAL_POINT_GROUNDING_SOURCE, UniversalActionError("本地落点证据来源无效。"))
        reject_if(self.scene_fingerprint != scene.fingerprint, UniversalActionError("本地落点证据不属于当前新鲜画面。"))
        reject_if(self.element_id != element.element_id or self.label != element.label, UniversalActionError("本地落点证据没有绑定当前唯一目标。"))
        reject_if(element.role not in {'button', 'icon', 'text', 'tab', 'toggle', 'image', 'list_item'}, UniversalActionError("当前目标类型不允许使用文字落点修正。"))
        reject_if(
            element.element_id.startswith('local_audited_') or element.meaning == 'application_text_input'
            or element.meaning.startswith(('input_', 'ime_', 'switch_keyboard_')),
            UniversalActionError("输入事务目标不允许使用普通文字落点修正。"),
        )
        model_bounds = self._normalized_numbers(self.model_bounds, 4, "模型目标框")
        grounded_bounds = self._normalized_numbers(self.grounded_bounds, 4, "本地文字框")
        proposed = self._normalized_numbers(self.proposed_point, 2, "模型提议落点")
        grounded = self._normalized_numbers(self.grounded_point, 2, "本地修正落点")
        reject_if(
            any((abs(actual - expected) > 1e-09 for actual, expected in zip(model_bounds,
            element.bounds))) or any((abs(actual - expected) > 1e-09 for actual, expected in zip(proposed,
            element.center))),
            UniversalActionError("本地落点证据与当前目标几何不一致。"),
        )
        left, top, right, bottom = grounded_bounds
        reject_if(not (left <= grounded[0] <= right and top <= grounded[1] <= bottom), UniversalActionError("本地修正落点不在已识别文字框内。"))
        reject_if(
            isinstance(self.matched_frames, bool) or isinstance(self.inspected_frames,
            bool) or (not isinstance(self.matched_frames, int)) or (not isinstance(self.inspected_frames,
            int)) or (not 2 <= self.matched_frames <= self.inspected_frames <= 4),
            UniversalActionError("本地落点证据缺少至少两帧稳定匹配。"),
        )
        model_left, model_top, model_right, model_bottom = model_bounds
        grounded_left, grounded_top, grounded_right, grounded_bottom = grounded_bounds
        horizontal_gap = max(0.0, grounded_left - model_right, model_left - grounded_right)
        vertical_gap = max(0.0, grounded_top - model_bottom, model_top - grounded_bottom)
        reject_if(math.hypot(horizontal_gap, vertical_gap) > MAX_LOCAL_GROUNDING_BOX_GAP, UniversalActionError("本地文字框与模型目标框不属于同一邻近区域。"))

@dataclass(frozen=True)
class ResolvedSemanticAction:
    """One device-independent action resolved from one fresh scene."""

    node_id: str
    kind: str
    normalized_point: NormalizedPoint | None = None
    normalized_end_point: NormalizedPoint | None = None
    text: str | None = None
    input_fragment: str | None = None
    input_method: str | None = None
    input_pinyin: str | None = None
    prior_input_value: str | None = None
    expected_input_value: str | None = None
    delete_count: int | None = None
    launch_ref: str | None = None
    expected_package_id: str | None = None
    target_app_id: str | None = None
    target_app_name: str | None = None
    direction: str | None = None
    hold_seconds: float | None = None
    path_distance: float | None = None
    target_element_id: str | None = None
    input_element_id: str | None = None
    destination_element_id: str | None = None
    before_fingerprint: str = ""
    expected_effect: dict[str, Any] = field(default_factory=dict)
    formal_candidate_id: str = ""
    formal_transition: dict[str, Any] = field(default_factory=dict)
    proposed_normalized_point: NormalizedPoint | None = None
    point_grounding: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        value = dataclass_wire(self)
        for key in ('proposed_normalized_point', 'point_grounding'):
            if value[key] is None:
                value.pop(key)
        return value


class UniversalActionController:
    """Resolve semantic actions without knowing WeChat, Douyin or any App UI."""

    def __init__(self, *, min_confidence: float=MIN_TARGET_CONFIDENCE) -> None:
        self.min_confidence = float(min_confidence)

    def resolve_one(self, action: SemanticAction, scene: UIScene, *, confirmed: bool=False,
        local_point_grounding: LocalPointGrounding | None=None) -> ResolvedSemanticAction:
        scene.validate()
        reject_if(
            local_point_grounding is not None and action.action not in {'tap_semantic', 'dismiss_overlay', 'double_tap',
            'long_press'},
            UniversalActionError('本地文字落点只能修正当前已选中的普通点按目标。'),
        )
        reject_if(not scene.stable, UniversalActionError("页面仍在变化，不能执行动作。"))
        if float(scene.confidence) < self.min_confidence:
            target_local_candidate = scene.unique_trusted_goal_element(min_confidence=self.min_confidence)
            action_element_id = str(action.params.get("element_id") or "").strip()
            reject_if(
                action.action not in {'tap_semantic', 'dismiss_overlay', 'input_verified_text', 'press_enter',
                'clear_verified_text', 'double_tap', 'long_press'} or target_local_candidate is None
                or target_local_candidate.element_id != action_element_id,
                UniversalActionError('页面整体置信度不足，且没有唯一可信的目标局部证据。'),
            )
        formal_candidate_id = str(action.params.get('formal_candidate_id') or '').strip()
        formal_transition = dict(action.params.get("formal_transition") or {})
        expected_effect = dict(action.params.get("expected_effect") or {})
        reject_if('system_ui' in expected_effect and action.action != 'reveal_system_navigation', UniversalActionError('结构化 system_ui 后置条件只允许用于系统导航栏唤出动作。'))

        def resolved(kind: str | None=None, **values: Any) -> ResolvedSemanticAction:
            return ResolvedSemanticAction(node_id=action.node_id, kind=kind or action.action,
                before_fingerprint=scene.fingerprint, expected_effect=expected_effect,
                formal_candidate_id=formal_candidate_id, formal_transition=formal_transition, **values)

        if action.action == 'tap_semantic':
            element = self._resolve_target(action, scene)
            if element.states.get('local_text_clear') is True:
                self._validate_local_text_clear(element, scene, formal=bool(formal_candidate_id))
            if (element.meaning in {'input_exact_literal_key', 'switch_keyboard_layout', 'switch_keyboard_case',
                'switch_keyboard_input_mode'}):
                self._validate_input_auxiliary_tap(element, scene, expected_effect, formal=bool(formal_candidate_id))
            if element.meaning == 'input_next_field_key':
                self._validate_next_field_tap(element, formal_transition, expected_effect,
                    formal=bool(formal_candidate_id))
            resolved = self._point_action(action, element, expected_effect, scene.fingerprint, scene=scene,
                local_point_grounding=local_point_grounding)
            if (element.meaning in {'ime_exact_candidate', 'input_exact_literal_key', 'switch_keyboard_layout',
                'switch_keyboard_case', 'switch_keyboard_input_mode'}):
                expected_element = expected_effect.get("element_state")
                expected_states = expected_element.get('states') if isinstance(expected_element, dict) else None
                expected_value = expected_states.get('value') if isinstance(expected_states, dict) else None
                reject_if(not isinstance(expected_value, str), UniversalActionError('输入辅助键缺少精确输入值后置条件。'))
                resolved = replace(resolved, prior_input_value=element.states.get('prior_input_value'),
                    expected_input_value=expected_value, input_element_id=element.states.get('input_element_id'))
            return resolved
        if action.action == 'press_enter':
            element = self._resolve_target(action, scene)
            reject_if(element.meaning != 'input_exact_enter_key', UniversalActionError('press_enter 必须绑定本地审计的唯一换行键。'))
            self._validate_input_auxiliary_tap(element, scene, expected_effect, formal=bool(formal_candidate_id))
            resolved = self._point_action(action, element, expected_effect, scene.fingerprint)
            return replace(resolved, prior_input_value=element.states.get('prior_input_value'),
                expected_input_value=element.states.get('expected_input_value'),
                input_element_id=element.states.get('input_element_id'))
        if action.action == 'dismiss_overlay':
            if not action.params.get('target'):
                action = SemanticAction(node_id=action.node_id, action=action.action, params={**action.params,
                    'target': 'close'})
            element = self._resolve_target(action, scene)
            return self._point_action(action, element, expected_effect, scene.fingerprint, scene=scene,
                local_point_grounding=local_point_grounding)
        if action.action == 'input_verified_text':
            text = str(action.params.get("text") or "")
            element = self._resolve_target(action, scene, required_role="input")
            reject_if(element.states.get('focused') is not True, UniversalActionError("文字输入前必须有当前画面证明输入框已聚焦。"))
            reject_if(element.states.get('keyboard_layout') != 'qwerty', UniversalActionError("精确文字输入要求当前画面确认 QWERTY 键盘。"))
            reject_if(not formal_candidate_id and element.states.get('goal_relevant') is not True, UniversalActionError("文字输入目标必须由当前画面证明与当前目标相关。"))
            try:
                input_step = plan_from_input_states(text, element.states)
            except (ValueError, VerifiedTextTransactionError) as exc:
                raise UniversalActionError(f"无法建立精确文字输入事务：{exc}") from exc
            reject_if(input_step is None, UniversalActionError("输入框已经逐字等于目标文字，不得重复输入。"))
            reject_if(input_step.kind == 'literal_key', UniversalActionError('下一分段需要独立可见的数字、空格或符号键审计，不能按字母键盘猜测。'))
            reject_if(element.states.get('keyboard_input_mode') != input_step.required_mode, UniversalActionError('当前键盘输入模式与下一确定性文字分段不一致。'))
            reject_if(
                input_step.required_case_mode and element.states.get('keyboard_case_mode') !=
                input_step.required_case_mode,
                UniversalActionError('当前键盘大小写状态与下一确定性英文分段不一致。'),
            )
            reject_if(element.states.get('ime_preedit_text'), UniversalActionError("当前仍有未完成的输入法组合，禁止继续键入。"))
            expected_states = {'value': input_step.current_text, 'ime_preedit_text': input_step.pinyin,
                'ime_exact_candidate_text': input_step.segment} if input_step.kind == 'chinese_pinyin' else {
                'value': input_step.expected_value}
            if ('element_state' not in expected_effect and input_step.kind == 'direct_latin'
                and (input_step.current_text == '') and is_direct_latin_segment(input_step.segment)):
                expected_effect['element_state'] = {'meaning': element.meaning, 'states': expected_states}
            reject_if(expected_effect.get('element_state') != {'meaning': element.meaning, 'states': expected_states}, UniversalActionError('精确文字输入的后置条件没有绑定下一确定性分段。'))
            self._require_unique_input(scene, element, formal=bool(formal_candidate_id), required_states={
                'keyboard_layout': 'qwerty', 'keyboard_input_mode': input_step.required_mode,
                **({'keyboard_case_mode': input_step.required_case_mode} if input_step.required_case_mode else {}),
            }, require_empty_preedit=True, error='精确文字输入要求当前画面只有一个符合安全条件的目标输入框。')
            return resolved('input_verified_text', normalized_point=element.center, text=text,
                input_fragment=input_step.segment, input_method=input_step.kind, input_pinyin=input_step.pinyin or None,
                prior_input_value=input_step.current_text, expected_input_value=input_step.expected_value,
                target_element_id=element.element_id)
        if action.action == 'clear_verified_text':
            element = self._resolve_target(action, scene, required_role="input")
            observed_value = element.states.get("value")
            observed_preedit = element.states.get("ime_preedit_text", "")
            reject_if(element.states.get('focused') is not True, UniversalActionError("清空文字前必须有当前画面证明输入框已聚焦。"))
            reject_if(not isinstance(observed_value, str), UniversalActionError("清空文字要求当前画面提供精确 states.value。"))
            reject_if(not isinstance(observed_preedit, str), UniversalActionError("清空文字的输入法预编辑状态格式无效。"))
            extra_delete_units = element.states.get("clear_extra_delete_units", 0)
            reject_if(
                isinstance(extra_delete_units, bool) or not isinstance(extra_delete_units,
                int) or (not 0 <= extra_delete_units <= 30),
                UniversalActionError("清空文字的额外视觉行退格单位无效。"),
            )
            reject_if(not observed_value and (not observed_preedit), UniversalActionError("清空文字要求应用值或输入法预编辑至少一项非空。"))
            delete_count = editable_character_count(observed_value) + editable_character_count(
                observed_preedit) + extra_delete_units
            reject_if(not 1 <= delete_count <= 100, UniversalActionError("清空文字的已验证字符数必须在1～100之间。"))
            reject_if(not formal_candidate_id and element.states.get('goal_relevant') is not True, UniversalActionError("清空文字目标必须由当前画面证明与当前目标相关。"))
            self._require_unique_input(scene, element, formal=bool(formal_candidate_id), require_text=True,
                error='清空文字要求当前画面只有一个符合安全条件的非空目标输入框。')
            expected_state = expected_effect.get("element_state")
            reject_if(not isinstance(expected_state, dict), UniversalActionError("清空文字必须声明输入框空值后置条件。"))
            reject_if(
                str(expected_state.get('meaning') or '') != element.meaning
                or expected_state.get('states') != {'value': ''},
                UniversalActionError("清空文字的后置条件必须精确绑定原输入框空值。"),
            )
            return resolved('clear_verified_text', normalized_point=element.center, text='', delete_count=delete_count,
                target_element_id=element.element_id)
        if action.action == 'double_tap':
            element = self._resolve_target(action, scene)
            self._validate_gesture_point(element.center, label="双击落点")
            self._require_visual_postcondition('double_tap', expected_effect, scene)
            return self._point_action(action, element, expected_effect, scene.fingerprint, scene=scene,
                local_point_grounding=local_point_grounding)
        if action.action == 'long_press':
            element = self._resolve_target(action, scene)
            self._validate_gesture_point(element.center, label="长按落点")
            duration_ms = action.params.get("duration_ms", 800)
            reject_if(isinstance(duration_ms, bool) or not isinstance(duration_ms, (int, float)), UniversalActionError("长按 duration_ms 格式无效。"))
            reject_if(not 500 <= float(duration_ms) <= 2000, UniversalActionError("长按 duration_ms 必须在500～2000之间。"))
            self._require_visual_postcondition('long_press', expected_effect, scene)
            resolved = self._point_action(action, element, expected_effect, scene.fingerprint, scene=scene,
                local_point_grounding=local_point_grounding)
            return replace(resolved, hold_seconds=float(duration_ms) / 1000.0)
        if action.action == 'drag':
            source = self._resolve_target(action, scene, prefix="source_")
            destination = self._resolve_target(action, scene, prefix='destination_')
            reject_if(source.element_id == destination.element_id, UniversalActionError("拖动起点和终点不能是同一元素。"))
            source_container_error = compact_drag_source_container_error(scene, source)
            reject_if(source_container_error, UniversalActionError(source_container_error))
            self._validate_gesture_point(source.center, label="拖动起点")
            self._validate_gesture_point(destination.center, label="拖动终点")
            distance = math.dist(source.center, destination.center)
            reject_if(not MIN_DRAG_DISTANCE <= distance <= MAX_DRAG_DISTANCE, UniversalActionError(f'拖动两端中心距离必须在{MIN_DRAG_DISTANCE:.2f}～{MAX_DRAG_DISTANCE:.2f}个归一化屏幕单位之间。'))
            self._require_visual_postcondition("drag", expected_effect, scene)
            return resolved('drag', normalized_point=source.center, normalized_end_point=destination.center,
                target_element_id=source.element_id, destination_element_id=destination.element_id,
                hold_seconds=DRAG_DURATION_SECONDS, path_distance=distance)
        if action.action == 'reveal_system_navigation':
            unexpected = set(action.params) - {'expected_effect', 'formal_candidate_id', 'formal_transition'}
            reject_if(unexpected, UniversalActionError('系统导航栏唤出动作不能携带坐标、方向、距离或其他参数。'))
            self._require_hidden_immersive_navigation(scene)
            reject_if(expected_effect != REVEAL_SYSTEM_NAVIGATION_EFFECT, UniversalActionError('系统导航栏唤出动作必须精确声明导航栏可见后置条件。'))
            return resolved("reveal_system_navigation")
        if action.action == 'launch_app':
            allowed = {'target_surface_id', 'target_app_id', 'target_app_name', 'launch_ref', 'expected_app_id',
                'expected_effect', 'formal_candidate_id', 'formal_transition'}
            reject_if(set(action.params) != allowed, UniversalActionError("App 直启动作包含协议外字段。"))
            launch_ref = str(action.params.get('launch_ref') or '').strip()
            expected_app_id = str(action.params.get('expected_app_id') or '').strip()
            target_app_id = str(action.params.get('target_app_id') or '').strip()
            target_app_name = str(action.params.get('target_app_name') or '').strip()
            reject_if(not launch_ref or not expected_app_id or not target_app_id or not target_app_name
                or expected_effect != {'app_id': target_app_id},
                UniversalActionError("App 直启没有绑定受信任引用、目标包和 typed App 视觉身份。"))
            reject_if(not str(action.params.get('target_surface_id') or '').strip(),
                UniversalActionError("App 直启缺少 target_surface_id。"))
            return resolved('launch_app', launch_ref=launch_ref, expected_package_id=expected_app_id,
                target_app_id=target_app_id,
                target_app_name=target_app_name)
        if action.action == 'swipe':
            direction = str(action.params.get("direction") or "").strip().lower()
            reject_if(direction not in {'up', 'down', 'left', 'right'}, UniversalActionError(f"不支持的滑动方向：{direction}"))
            element_id = str(action.params.get("element_id") or "").strip()
            absence = expected_effect.get("element_absent")
            if element_id:
                element = self._resolve_target(action, scene)
                self._validate_targeted_swipe_absence_contract(element, absence)
                start, end = self._targeted_swipe_path(element, direction)
                distance = math.dist(start, end)
                return resolved('swipe', normalized_point=start, normalized_end_point=end, direction=direction,
                    hold_seconds=DRAG_DURATION_SECONDS, path_distance=distance, target_element_id=element.element_id)
            reject_if(absence is not None, UniversalActionError('元素消失后置条件必须绑定同一 swipe element_id。'))
            return resolved("swipe", direction=direction)
        if action.action in {'back', 'home', 'open_recent_apps', 'wait_for_change'}:
            return resolved()
        raise UniversalActionError(f"通用动作控制器尚不支持：{action.action}")

    def _require_unique_input(self, scene: UIScene, target: UIElement, *, formal: bool,
        required_states: Mapping[str, Any] | None=None, require_empty_preedit: bool=False,
        require_text: bool=False, error: str) -> None:
        required_states = required_states or {}
        candidates = tuple(element for element in self._visible_elements(scene, role='input') if
            (element.element_id == target.element_id if formal else element.states.get('goal_relevant') is True)
            and element.states.get('focused') is True
            and all(element.states.get(key) == value for key, value in required_states.items())
            and (not require_empty_preedit or not element.states.get('ime_preedit_text'))
            and (not require_text or (isinstance(element.states.get('value'), str)
                and isinstance(element.states.get('ime_preedit_text', ''), str)
                and bool(element.states.get('value') or element.states.get('ime_preedit_text')))))
        reject_if(len(candidates) != 1 or candidates[0].element_id != target.element_id,
            UniversalActionError(error))

    def _validate_local_text_clear(self, element: UIElement, scene: UIScene, *, formal: bool=False) -> None:
        reject_if(
            element.meaning != 'clear_local_text' or element.role not in {'button',
            'icon'} or element.label.strip().casefold() not in {'×', '✕', '✖', 'x'},
            UniversalActionError("本地文字清空必须绑定真实可见的独立 × 图形。"),
        )
        inputs = tuple((candidate for candidate in scene.elements if candidate.role == 'input'
            and float(candidate.confidence) >= self.min_confidence and (formal or candidate.states.get('goal_relevant')
            is True) and (candidate.states.get('focused') is True) and isinstance(candidate.states.get('value'),
            str) and bool(candidate.states.get('value')) and (candidate.states.get('keyboard_layout') in {'qwerty',
            'numeric', 'symbol', 'unknown'})))
        reject_if(len(inputs) != 1, UniversalActionError('本地文字清空要求唯一非空、已聚焦且带软键盘事实的目标输入框。'))
        input_element = inputs[0]
        il, it, ir, ib = input_element.bounds
        el, et, er, eb = element.bounds
        element_height = max(1e-9, eb - et)
        input_height = max(1e-9, ib - it)
        vertical_overlap = max(0.0, min(ib, eb) - max(it, et))
        reject_if(
            not (vertical_overlap / element_height >= 0.6 and el >= il + 0.4 * (ir - il) and (er <= min(1.0,
            ir + 0.2)) and (max(0.0, el - ir) <= max(0.04, input_height)) and (er - el <= 2.0 * input_height)
            and (element_height <= 1.5 * input_height)),
            UniversalActionError("本地文字清空控件没有与唯一目标输入框形成可信几何绑定。"),
        )

    def _validate_next_field_tap(self, element: UIElement, formal_transition: dict[str, Any], expected_effect: dict[str,
        Any], *, formal: bool) -> None:
        states = element.states
        source_id = str(states.get("source_input_field_id") or "").strip()
        target_id = str(states.get("target_input_field_id") or "").strip()
        target_label = str(states.get("target_input_field_label") or "").strip()
        expectations = formal_transition.get("expectations")
        reject_if(
            not (formal and element.role == 'button' and (float(element.confidence) >= 0.9)
            and (states.get('fully_visible') is True) and (states.get('input_next_field_key') is True)
            and (states.get('key_action') == 'next') and source_id and target_id and target_label
            and (source_id != target_id) and (expected_effect == {'scene_changed': True}) and isinstance(expectations,
            list) and (expectations == [{'subject_ref': target_id, 'predicate': 'input_field.focused',
            'operator': 'equals', 'value': True}])),
            UniversalActionError("Next键没有绑定唯一typed字段切换合同。"),
        )

    def _validate_input_auxiliary_tap(self, element: UIElement, scene: UIScene, expected_effect: dict[str, Any], *,
        formal: bool=False) -> None:
        states = element.states
        reject_if(
            element.role != 'button' or float(element.confidence) < 0.9 or (not formal and states.get('goal_relevant')
            is not True) or (states.get('fully_visible') is not True),
            UniversalActionError("输入辅助键缺少本轮完整、高置信本地审计。"),
        )
        input_id = str(states.get("input_element_id") or "").strip()
        input_element = self._required_element(scene, input_id, error='输入辅助键没有绑定唯一输入框')
        prior_value = states.get("prior_input_value")
        reject_if(
            input_element.role != 'input' or input_element.states.get('focused') is not True
            or (not isinstance(prior_value, str)) or (input_element.states.get('value') != prior_value),
            UniversalActionError("输入辅助键与当前精确输入前缀不一致。"),
        )
        expected_state = expected_effect.get("element_state")
        reject_if(not isinstance(expected_state, dict) or expected_state.get('meaning') != input_element.meaning, UniversalActionError("输入辅助键没有绑定原输入框的后置条件。"))
        expected_states = expected_state.get("states")
        reject_if(not isinstance(expected_states, dict), UniversalActionError("输入辅助键后置状态格式无效。"))
        if element.meaning == 'input_exact_literal_key':
            key_value = states.get("key_value")
            expected_value = states.get("expected_input_value")
            reject_if(
                states.get('input_literal_key') is not True or not isinstance(key_value,
                str) or len(key_value) != 1 or (expected_value != prior_value + key_value)
                or (expected_states != {'value': expected_value}),
                UniversalActionError("逐键输入没有绑定唯一下一字符和精确结果。"),
            )
        elif element.meaning == 'input_exact_enter_key':
            key_value = states.get("key_value")
            expected_value = states.get("expected_input_value")
            reject_if(
                states.get('input_enter_key') is not True or states.get('key_action') != 'newline' or key_value != '\n'
                or (expected_value != prior_value + '\n') or (expected_states != {'value': expected_value}),
                UniversalActionError('换行键没有绑定 multiline newline 与精确输入结果。'),
            )
            reject_if(input_element.states.get('input_multiline') is not True, UniversalActionError("当前输入框没有本地多行字段凭据。"))
        elif element.meaning in {'switch_keyboard_layout', 'switch_keyboard_case', 'switch_keyboard_input_mode'}:
            specs = {
                'switch_keyboard_layout': ('keyboard_layout_switch', 'current_layout', 'target_layout',
                    'keyboard_layout', {'qwerty', 'numeric', 'symbol'}, {}, '键盘布局切换方向或后置条件无效。'),
                'switch_keyboard_case': ('keyboard_case_switch', 'current_mode', 'target_mode',
                    'keyboard_case_mode', {'lower', 'upper'}, {'keyboard_layout': 'qwerty',
                    'keyboard_input_mode': 'direct_latin'}, '键盘大小写切换方向或后置条件无效。'),
                'switch_keyboard_input_mode': ('keyboard_input_mode_switch', 'current_mode', 'target_mode',
                    'keyboard_input_mode', {'direct_latin', 'chinese_pinyin'}, {'keyboard_layout': 'qwerty'},
                    '键盘输入模式切换方向或后置条件无效。'),
            }
            flag, current_key, target_key, state_key, modes, fixed_states, message = specs[element.meaning]
            current, target = states.get(current_key), states.get(target_key)
            required_input_states = {**fixed_states, state_key: current}
            reject_if(states.get(flag) is not True or current not in modes or target not in modes
                or current == target or any(input_element.states.get(key) != value for key,
                value in required_input_states.items()) or expected_states != {'value': prior_value,
                state_key: target}, UniversalActionError(message))

    def verify_after_action(self, resolved: ResolvedSemanticAction, before: UIScene, after: UIScene) -> tuple[str, ...]:
        before.validate()
        after.validate()
        if resolved.kind != 'wait_for_change':
            reject_if(not resolved.before_fingerprint, UniversalActionError("动作缺少执行前场景 fingerprint。"))
            reject_if(resolved.before_fingerprint != before.fingerprint, UniversalActionError("动作绑定的 fingerprint 已过期。"))
        if resolved.kind in {'double_tap', 'long_press', 'drag'}:
            self._require_visual_postcondition(resolved.kind, resolved.expected_effect, before)
        if resolved.kind == 'long_press':
            self._verify_long_press_contract(resolved, before)
        reject_if(not after.stable or float(after.confidence) < self.min_confidence, UniversalActionError("动作后的页面不稳定或置信度不足。"))
        reject_if(
            resolved.kind != 'wait_for_change' and resolved.kind != 'reveal_system_navigation'
            and (resolved.expected_effect.get('allow_unchanged') is not True) and (before.fingerprint
            and after.fingerprint and (before.fingerprint == after.fingerprint) or (resolved.kind != 'drag'
            and self.scenes_semantically_equivalent(before, after))),
            UniversalActionError("动作后页面没有可验证的语义变化。"),
        )
        expected = resolved.expected_effect
        expected_app = str(expected.get("app_id") or "").strip()
        if resolved.kind == 'launch_app':
            observed_app = str(after.foreground_app_id or '').strip().casefold()
            package_match = bool(resolved.expected_package_id and observed_app == resolved.expected_package_id.strip(
                ).casefold())
            reject_if(not package_match and not scene_matches_app_identity(after, resolved.target_app_id or '',
                resolved.target_app_name or ''), UniversalActionError(f'动作后画面不能证明目标 App 已在前台：{after.foreground_app_id}。'))
        else:
            reject_if(expected_app and after.foreground_app_id != expected_app, UniversalActionError(f'动作后前台 App 不符合预期：{after.foreground_app_id} != {expected_app}'))
        expected_screen = str(expected.get("screen_id") or "").strip()
        reject_if(expected_screen and after.screen_id != expected_screen, UniversalActionError(f'动作后页面不符合预期：{after.screen_id} != {expected_screen}'))
        input_preedit_pending = False
        if resolved.kind in {'input_verified_text', 'press_enter', 'clear_verified_text'}:
            input_preedit_pending = self._verify_exact_input_value(resolved, before, after)
        if expected.get('element_absent') is not None:
            self._verify_expected_element_absent(resolved, before, after)
        if resolved.formal_candidate_id:
            self._verify_formal_transition(resolved, before, after)
        element_state = expected.get("element_state")
        if element_state is not None:
            reject_if(not isinstance(element_state, dict), UniversalActionError("expected_effect.element_state 格式无效。"))
            meaning = str(element_state.get("meaning") or "").strip()
            states = dict(element_state.get("states") or {})
            try:
                after.resolve_unique(meaning=meaning, states=states, min_confidence=self.min_confidence)
            except UISceneError as exc:
                before_target_is_input = False
                if resolved.target_element_id:
                    try:
                        before_target_is_input = before.get_element(resolved.target_element_id).role == 'input'
                    except UISceneError:
                        before_target_is_input = False
                input_aliases = tuple((element for element in after.elements if resolved.kind in {'tap_semantic',
                    'input_verified_text', 'clear_verified_text'} and before_target_is_input
                    and (element.role == 'input') and (float(element.confidence) >= self.min_confidence)
                    and all((element.states.get(key) == value for key, value in states.items()))))
                provisional_direct_preedits = tuple((element for element
                    in after.elements if resolved.kind == 'input_verified_text'
                    and resolved.input_method == 'direct_latin' and before_target_is_input
                    and (element.role == 'input') and (float(element.confidence) >= self.min_confidence)
                    and self._is_exact_direct_latin_preedit_transition(resolved, after, element)))
                if len(input_aliases) != 1 and len(provisional_direct_preedits) != 1:
                    raise UniversalActionError(f'动作结果缺少元素状态证据：{exc}') from exc
        if resolved.kind == 'long_press':
            self._verify_long_press_result(resolved, before, after)
        if resolved.kind == 'drag':
            self._verify_drag_result(resolved, before, after)
        if resolved.kind == 'reveal_system_navigation':
            self._verify_revealed_system_navigation(resolved, before, after)
        evidence: list[str] = []
        if any((expected.get(key) is True for key in ('scene_changed', 'content_changed',
            'current_video_changed'))):
            evidence.append(f'控制器确认动作前后场景指纹发生变化：{before.fingerprint} -> {after.fingerprint}')
        if expected_app:
            evidence.append(f"控制器确认前台 App 为 {expected_app}")
        if expected_screen:
            evidence.append(f"控制器确认页面为 {expected_screen}")
        if element_state is not None:
            evidence.append(CONTROLLER_INPUT_PREEDIT_PENDING if input_preedit_pending else
                f"控制器确认目标元素状态：{str(element_state.get('meaning') or '').strip()} "
                f"{dict(element_state.get('states') or {})}")
        element_absent = expected.get("element_absent")
        if isinstance(element_absent, Mapping):
            identity = str(element_absent.get('label') or '').strip() or str(element_absent.get('meaning')
                or '').strip() or str(element_absent.get('element_id') or '').strip()
            evidence.append(f"控制器确认目标元素已消失：{identity}")
        return tuple(evidence)

    @staticmethod
    def _require_hidden_immersive_navigation(scene: UIScene) -> Any:
        system_ui = scene.system_ui
        reject_if(system_ui.immersive_or_fullscreen is not True or system_ui.navigation_bar_visible is not False, UniversalActionError('系统导航栏唤出动作要求当前画面明确处于沉浸态且导航栏隐藏。'))
        return system_ui

    def _verify_revealed_system_navigation(self, resolved: ResolvedSemanticAction, before: UIScene,
        after: UIScene) -> None:
        self._require_hidden_immersive_navigation(before)
        reject_if(resolved.expected_effect != REVEAL_SYSTEM_NAVIGATION_EFFECT, UniversalActionError("系统导航栏唤出动作的结构化后置条件无效。"))
        reject_if(after.system_ui.navigation_bar_visible is not True, UniversalActionError("动作后缺少结构化导航栏可见证据。"))

    @staticmethod
    def _validate_gesture_point(point: NormalizedPoint, *, label: str) -> None:
        x, y = point
        reject_if(
            not (GESTURE_EDGE_MARGIN <= x <= 1.0 - GESTURE_EDGE_MARGIN
            and GESTURE_EDGE_MARGIN <= y <= 1.0 - GESTURE_EDGE_MARGIN),
            UniversalActionError(f'{label}必须离画面边缘至少{GESTURE_EDGE_MARGIN:.2f}个归一化屏幕单位。'),
        )

    @staticmethod
    def _validate_targeted_swipe_absence_contract(element: UIElement, absence: Any) -> None:
        reject_if(not isinstance(absence, Mapping) or set(absence) != {'element_id', 'meaning', 'role', 'label'}, UniversalActionError('元素绑定滑动必须精确声明同一目标的 element_absent 后置条件。'))
        expected = {'element_id': element.element_id, 'meaning': element.meaning, 'role': element.role,
            'label': element.label}
        reject_if(dict(absence) != expected, UniversalActionError('元素绑定滑动的消失目标与当前可信元素不一致。'))

    @classmethod
    def _targeted_swipe_path(cls, element: UIElement, direction: str) -> tuple[NormalizedPoint, NormalizedPoint]:
        left, top, right, bottom = element.bounds
        width = right - left
        height = bottom - top
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        if direction == 'up':
            start = (center_x, top + height * 0.75)
            end = (center_x, TARGETED_SWIPE_EDGE_MARGIN)
        elif direction == 'down':
            start = (center_x, top + height * 0.25)
            end = (center_x, 1.0 - TARGETED_SWIPE_EDGE_MARGIN)
        elif direction == 'left':
            start = (left + width * 0.75, center_y)
            end = (TARGETED_SWIPE_EDGE_MARGIN, center_y)
        elif direction == 'right':
            start = (left + width * 0.25, center_y)
            end = (1.0 - TARGETED_SWIPE_EDGE_MARGIN, center_y)
        else:
            raise UniversalActionError(f"不支持的元素滑动方向：{direction}")
        cls._validate_gesture_point(start, label="元素滑动起点")
        cls._validate_gesture_point(end, label="元素滑动终点")
        delta_x = end[0] - start[0]
        delta_y = end[1] - start[1]
        direction_matches = {'up': delta_y < 0 and abs(delta_y) > abs(delta_x),
            'down': delta_y > 0 and abs(delta_y) > abs(delta_x), 'left': delta_x < 0 and abs(delta_x) > abs(delta_y),
            'right': delta_x > 0 and abs(delta_x) > abs(delta_y)}[direction]
        distance = math.dist(start, end)
        reject_if(not direction_matches, UniversalActionError("元素边界无法形成指定方向的滑动轨迹。"))
        reject_if(not MIN_DRAG_DISTANCE <= distance <= MAX_DRAG_DISTANCE, UniversalActionError(f'元素滑动轨迹距离必须在{MIN_DRAG_DISTANCE:.2f}～{MAX_DRAG_DISTANCE:.2f}之间。'))
        return start, end

    @staticmethod
    def _element_identity_surface_is_continuous(before: UIScene, after: UIScene) -> bool:
        """Carry an observation-local element identity only across a continuous typed surface."""

        if scene_surface_kind(before) != scene_surface_kind(after):
            return False
        before_app = before.foreground_app_id.strip().casefold()
        after_app = after.foreground_app_id.strip().casefold()
        if before_app not in {'', 'unknown'} and after_app not in {'', 'unknown'} and (before_app != after_app):
            return False
        before_screen = before.screen_id.strip().casefold()
        after_screen = after.screen_id.strip().casefold()
        if (before_screen not in {'', 'unknown'} and after_screen not in {'',
            'unknown'} and (before_screen != after_screen)):
            return False
        return True

    @staticmethod
    def _regions_stably_overlap(before_bounds: NormalizedBounds, after_bounds: NormalizedBounds) -> bool:
        return bounds_overlap(before_bounds, after_bounds)['intersection_over_smaller'] >= 0.60

    def _visible_elements(self, scene: UIScene, *, role: str | None=None,
        element_id: str | None=None) -> tuple[UIElement, ...]:
        return tuple(element for element in scene.elements if (role is None or element.role == role)
            and (element_id is None or element.element_id == element_id)
            and float(element.confidence) >= self.min_confidence and element.states.get('visible') is not False)

    def _matching_after_elements(self, before_element: UIElement, after: UIScene, *,
        overlap_fallback: bool=False) -> tuple[UIElement, ...]:
        visible = self._visible_elements(after, role=before_element.role)
        matches = tuple(element for element in visible if element.element_id == before_element.element_id)
        if matches:
            return matches
        matches = tuple(element for element in visible if element.meaning.casefold() ==
            before_element.meaning.casefold() and element.label.casefold() == before_element.label.casefold())
        if overlap_fallback and len(matches) != 1:
            matches = tuple(element for element in visible if self._regions_stably_overlap(
                before_element.bounds, element.bounds))
        return matches

    def _required_element(self, scene: UIScene, element_id: str, *, error: str) -> UIElement:
        try:
            return scene.get_element(element_id, min_confidence=self.min_confidence)
        except UISceneError as exc:
            raise UniversalActionError(f'{error}：{exc}') from exc

    @classmethod
    def _is_same_absence_target(cls, before_target: UIElement, after_element: UIElement, *,
        surface_is_continuous: bool) -> bool:
        before_label = before_target.label.strip().casefold()
        after_label = after_element.label.strip().casefold()
        before_role = before_target.role.strip().casefold()
        after_role = after_element.role.strip().casefold()
        before_meaning = before_target.meaning.strip().casefold()
        after_meaning = after_element.meaning.strip().casefold()
        label_matches = bool(before_label) and before_label == after_label
        exact_semantics = bool(before_meaning) and before_role == after_role and (before_meaning ==
            after_meaning) and (label_matches or not before_label or (not after_label))
        if exact_semantics:
            return True
        if not surface_is_continuous:
            return False
        spatially_continuous = cls._regions_stably_overlap(before_target.bounds, after_element.bounds)
        return spatially_continuous and (after_element.element_id == before_target.element_id or label_matches)

    def _verify_expected_element_absent(self, resolved: ResolvedSemanticAction, before: UIScene,
        after: UIScene) -> None:
        absence = resolved.expected_effect.get("element_absent")
        target_id = str(resolved.target_element_id or "").strip()
        reject_if(resolved.kind != 'swipe' or not target_id, UniversalActionError('element_absent 结果没有绑定元素滑动动作。'))
        before_target = self._required_element(before, target_id, error='元素滑动前目标证据无效')
        self._validate_targeted_swipe_absence_contract(before_target, absence)
        surface_is_continuous = self._element_identity_surface_is_continuous(before, after)
        still_visible = tuple((element for element in self._visible_elements(after) if
            self._is_same_absence_target(before_target, element, surface_is_continuous=surface_is_continuous)))
        reject_if(still_visible, UniversalActionError('元素滑动后同一目标仍然可见，不能判定已划掉。'))

    @staticmethod
    def _has_structured_postcondition(expected: dict[str, Any], before: UIScene) -> bool:
        if (any((expected.get(key) is True for key in ('scene_changed', 'content_changed',
            'current_video_changed'))) or UniversalActionController._expects_surface_change(expected, before)):
            return True
        element_state = expected.get("element_state")
        element_absent = expected.get("element_absent")
        if isinstance(element_absent, Mapping):
            return bool(str(element_absent.get("element_id") or "").strip())
        return bool(isinstance(element_state, dict) and str(element_state.get('meaning') or '').strip()
            and isinstance(element_state.get('states'), dict) and element_state['states'])

    @staticmethod
    def _expects_surface_change(expected: dict[str, Any], before: UIScene) -> bool:
        expected_app = str(expected.get("app_id") or "").strip()
        if expected_app and expected_app != before.foreground_app_id:
            return True
        expected_screen = str(expected.get("screen_id") or "").strip()
        return bool(expected_screen and expected_screen != before.screen_id)

    @classmethod
    def _require_visual_postcondition(cls, kind: str, expected: dict[str, Any], before: UIScene) -> None:
        reject_if(expected.get('allow_unchanged') is True, UniversalActionError(f"{kind} 禁止声明 allow_unchanged。"))
        reject_if(not cls._has_structured_postcondition(expected, before), UniversalActionError(f'{kind} 必须声明可由动作后新画面验证的结构化预期。'))

    def _verify_drag_result(self, resolved: ResolvedSemanticAction, before: UIScene, after: UIScene) -> None:
        source_id = str(resolved.target_element_id or "").strip()
        destination_id = str(resolved.destination_element_id or "").strip()
        reject_if(not source_id or not destination_id, UniversalActionError("拖动结果缺少起点或终点元素身份。"))
        source = self._required_element(before, source_id, error='拖动前端点证据无效')
        destination = self._required_element(before, destination_id, error='拖动前端点证据无效')

        source_container_error = compact_drag_source_container_error(before, source)
        reject_if(source_container_error, UniversalActionError(source_container_error))
        self._validate_gesture_point(source.center, label="拖动起点")
        self._validate_gesture_point(destination.center, label="拖动终点")

        distance = math.dist(source.center, destination.center)
        reject_if(not MIN_DRAG_DISTANCE <= distance <= MAX_DRAG_DISTANCE, UniversalActionError("拖动路径距离超出安全范围。"))
        reject_if(resolved.path_distance is None or abs(float(resolved.path_distance) - distance) > 1e-06, UniversalActionError("拖动路径距离与动作前端点不一致。"))
        reject_if(resolved.hold_seconds != DRAG_DURATION_SECONDS, UniversalActionError("拖动执行时长不是控制器固定的0.8秒。"))

        candidates = self._matching_after_elements(source, after)
        if len(candidates) == 1:
            moved = math.dist(source.center, candidates[0].center)
            remaining = math.dist(candidates[0].center, destination.center)
            required_improvement = max(0.03, distance * 0.20)
            if moved >= MIN_DRAG_RESULT_DISPLACEMENT and remaining <= distance - required_improvement:
                return

        expected = resolved.expected_effect
        has_alternative_proof = bool(self._element_state_transition_expected(expected, before)
            or self._expects_surface_change(expected, before))
        reject_if(not has_alternative_proof, UniversalActionError('拖动后缺少源元素向终点显著移动或等价结构化状态证据。'))

    def _verify_long_press_result(self, resolved: ResolvedSemanticAction, before: UIScene, after: UIScene) -> None:
        if set(after.overlays) - set(before.overlays):
            return
        target_id = str(resolved.target_element_id or "").strip()
        before_target = self._required_element(before, target_id, error='长按前目标证据无效')
        after_targets = self._visible_elements(after, role=before_target.role, element_id=target_id)
        if len(after_targets) == 1 and after_targets[0].states != before_target.states:
            return
        result_markers = ('verification', 'status', 'result', 'outcome', 'feedback', '验证', '状态', '结果', '反馈')
        new_structured_results = tuple((element for element in after.elements if element.role in {'text', 'button',
            'icon'} and element.states.get('goal_relevant') is True and (element.states.get('fully_visible') is True)
            and (float(element.confidence) >= self.min_confidence) and bool(str(element.label or '').strip())
            and bool(element.evidence) and any((marker in str(element.meaning or '').casefold() for marker
            in result_markers)) and (not any((prior.role == element.role and prior.meaning == element.meaning
            and (prior.label == element.label) for prior in before.elements)))))
        if len(new_structured_results) == 1:
            return
        expected = resolved.expected_effect
        if (self._element_state_transition_expected(expected, before)
            or self._expects_surface_change(expected, before)):
            return
        raise UniversalActionError('长按后缺少新增弹层、目标状态变化或等价结构化结果证据。')

    def _verify_formal_transition(self, resolved: ResolvedSemanticAction, before: UIScene, after: UIScene) -> None:
        transition = resolved.formal_transition
        expectations = transition.get("expectations") if isinstance(transition, dict) else None
        reject_if(not isinstance(expectations, list) or not expectations, UniversalActionError("正式候选缺少 typed transition expectations。"))
        for expectation in expectations:
            reject_if(not isinstance(expectation, dict), UniversalActionError("typed transition expectation 格式无效。"))
            predicate = str(expectation.get("predicate") or "")
            operator = str(expectation.get("operator") or "")
            value = expectation.get("value")
            key = (predicate, operator)
            if key == ('surface.kind', 'equals'):
                actual = scene_surface_kind(after)
                reject_if(actual != value, UniversalActionError("typed surface.kind 后置状态未满足。"))
            elif key == ('surface.overlay_present', 'equals'):
                reject_if(bool(after.overlays) is not bool(value), UniversalActionError("typed overlay 后置状态未满足。"))
            elif key == ('element.exists', 'absent'):
                self._verify_expected_element_absent(resolved, before, after)
            elif key == ('system_ui.navigation_bar_visible', 'equals'):
                reject_if(after.system_ui.navigation_bar_visible is not value, UniversalActionError("typed system_ui 后置状态未满足。"))
            elif key == ('element.state.value', 'equals'):
                expected_transition_value = (
                    resolved.prior_input_value
                    if resolved.kind == "input_verified_text" and resolved.input_method == "chinese_pinyin"
                    else resolved.expected_input_value
                )
                reject_if(expected_transition_value != value and resolved.text != value, UniversalActionError("typed input value 与已验证事务不一致。"))
            elif key == ('element.state.ime_preedit_text', 'equals'):
                reject_if(
                    resolved.kind != 'input_verified_text' or resolved.input_method != 'chinese_pinyin'
                    or resolved.input_pinyin != value,
                    UniversalActionError("typed 拼音组合状态与已验证中文事务不一致。"),
                )
            elif key == ('element.state.ime_preedit_text', 'absent'):
                reject_if(resolved.kind != 'clear_verified_text', UniversalActionError("typed 预编辑清空后置状态未绑定清空动作。"))
            elif key == ('element.state.ime_exact_candidate_text', 'equals'):
                reject_if(
                    resolved.kind != 'input_verified_text' or resolved.input_method != 'chinese_pinyin'
                    or resolved.input_fragment != value,
                    UniversalActionError("typed 中文候选状态与已验证中文事务不一致。"),
                )
            elif (key in {('element.state.keyboard_layout', 'equals'), ('element.state.keyboard_input_mode', 'equals'),
                ('element.state.keyboard_case_mode', 'equals')}):
                state_key = predicate.removeprefix("element.state.")
                expected_element = resolved.expected_effect.get("element_state")
                expected_states = expected_element.get("states") if isinstance(expected_element, dict) else None
                input_element_id = str(resolved.input_element_id or "").strip()
                reject_if(
                    resolved.kind != 'tap_semantic' or not input_element_id or (not isinstance(value,
                    str)) or (not value) or (not isinstance(expected_states,
                    dict)) or (expected_states.get(state_key) != value)
                    or (expected_states.get('value') != resolved.expected_input_value),
                    UniversalActionError(f"typed {state_key} 与已验证输入辅助动作不一致。"),
                )
                match_count = sum((1 for item in after.elements if item.element_id == input_element_id
                    and item.role == 'input' and (item.states.get(state_key) == value)
                    and (item.states.get('value') == resolved.expected_input_value)
                    and (float(item.confidence) >= MIN_TARGET_CONFIDENCE) and (item.states.get('visible')
                    is not False)))
                reject_if(match_count != 1, UniversalActionError(f"typed {state_key} 后置状态未满足。"))
            elif key == ('effect.applied', 'equals'):
                reject_if(value is not True or before.fingerprint == after.fingerprint, UniversalActionError("typed effect receipt 缺少动作后变化证据。"))
            elif key in {('surface.active_ref', 'equals'), ('surface.focused_entity_ref', 'equals')}:
                reject_if(before.fingerprint == after.fingerprint, UniversalActionError("typed surface 目标没有产生新观察。"))
            elif (key in {('surface.navigation_depth', 'changed'), ('surface.viewport', 'changed'),
                ('observation.changed', 'changed'), ('scene.changed', 'changed'), ('element.state.interaction_result',
                'changed'), ('element.state.location_relation', 'changed')}):
                reject_if(before.fingerprint == after.fingerprint, UniversalActionError("typed changed 后置状态未满足。"))
            elif key == ('element.state.focused', 'equals'):
                target_id = str(resolved.target_element_id or "")
                matches = [item for item in after.elements if item.element_id == target_id]
                if len(matches) == 1 and matches[0].states.get('focused') is value:
                    continue
                focus_only_source_count = sum((1 for item in before.elements if item.element_id == target_id
                    and item.role == 'input' and (item.states.get('focus_only_input_surface') is True)))
                audited_focus_count = sum((1 for item in after.elements if item.role == 'input'
                    and item.meaning == 'application_text_input' and item.element_id.startswith('local_audited_')
                    and (item.states.get('focused') is value) and (item.states.get('fully_visible') is True)
                    and (item.states.get('primary_input_geometry_verified') is True)
                    and (item.states.get('geometry_audit_source') == 'input_structure_audit')
                    and (str(item.states.get('input_field_id') or '').strip() not in {'',
                    'unknown'}) and (float(item.confidence) >= self.min_confidence)))
                reject_if(
                    not (value is True and focus_only_source_count == audited_focus_count == 1
                    and input_app_identity_compatible(before.foreground_app_id,
                    after.foreground_app_id) and input_screen_identity_compatible(before.screen_id, after.screen_id)),
                    UniversalActionError("typed focused 后置状态未满足。"),
                )
            elif key == ('input_field.focused', 'equals'):
                executed_targets = [item for item in before.elements if item.element_id == resolved.target_element_id
                    and item.meaning == 'input_next_field_key']
                target_label = str(executed_targets[0].states.get('target_input_field_label')
                    or '').strip() if len(executed_targets) == 1 else ''
                match_count = sum((1 for item in after.elements if item.role == 'input'
                    and item.states.get('input_field_id') == expectation.get('subject_ref')
                    and (item.states.get('input_field_label') == target_label) and (item.states.get('focused')
                    is value) and (float(item.confidence) >= MIN_TARGET_CONFIDENCE)))
                reject_if(value is not True or match_count != 1, UniversalActionError("typed目标字段聚焦后置状态未满足。"))
            elif key == ('element.state.location_relation', 'equals'):
                reject_if(
                    resolved.kind != 'drag' or not resolved.target_element_id or (not resolved.destination_element_id)
                    or (not isinstance(value, str)) or (not value),
                    UniversalActionError("typed drag relation 与已解析动作不一致。"),
                )
            else:
                raise UniversalActionError(f'尚未实现的 typed transition expectation：{predicate}/{operator}')

    def _element_state_transition_expected(self, expected: dict[str, Any], before: UIScene) -> bool:
        element_state = expected.get("element_state")
        if not isinstance(element_state, dict):
            return False
        meaning = str(element_state.get("meaning") or "").strip()
        states = element_state.get("states")
        if not meaning or not isinstance(states, dict) or (not states):
            return False
        matches = tuple((element for element in self._visible_elements(before) if element.meaning == meaning
            and all((element.states.get(key) == value for key, value in states.items()))))
        return not matches

    def _verify_long_press_contract(self, resolved: ResolvedSemanticAction, before: UIScene) -> None:
        target_id = str(resolved.target_element_id or "").strip()
        reject_if(not target_id or resolved.normalized_point is None, UniversalActionError("长按结果缺少目标元素身份或落点。"))
        reject_if(resolved.hold_seconds is None or not 0.5 <= float(resolved.hold_seconds) <= 2.0, UniversalActionError("长按执行时长必须在0.5～2.0秒之间。"))
        target = self._required_element(before, target_id, error='长按前目标证据无效')
        reject_if(target.role == 'container', UniversalActionError("页面容器不是可长按控件。"))
        self._validate_gesture_point(target.center, label="长按落点")
        reject_if(math.dist(resolved.normalized_point, target.center) > 1e-06, UniversalActionError("长按落点与动作前目标中心不一致。"))

    def _verify_exact_input_value(self, resolved: ResolvedSemanticAction, before: UIScene, after: UIScene) -> bool:
        expected = resolved.expected_input_value if resolved.kind in {'input_verified_text',
            'press_enter'} else resolved.text
        target_id = str(resolved.input_element_id or resolved.target_element_id or '').strip()
        reject_if(expected is None or not target_id, UniversalActionError("输入动作缺少精确文字或目标输入框身份。"))
        reject_if(
            resolved.kind == 'input_verified_text' and (not expected or not resolved.input_fragment
            or (not resolved.input_method)),
            UniversalActionError("输入动作缺少精确文字或目标输入框身份。"),
        )
        reject_if(
            resolved.kind == 'press_enter' and (resolved.prior_input_value is None
            or expected != resolved.prior_input_value + '\n'),
            UniversalActionError("换行动作缺少精确前缀或 newline 后置值。"),
        )
        if resolved.kind == 'clear_verified_text':
            reject_if(expected != '' or resolved.delete_count is None, UniversalActionError("清空动作缺少空值或精确退格次数。"))
        before_input = self._required_element(before, target_id, error='输入前目标证据无效')
        reject_if(before_input.role != 'input', UniversalActionError("输入前目标不是 input 元素。"))
        reject_if(not resolved.formal_candidate_id and before_input.states.get('goal_relevant') is not True, UniversalActionError("输入前目标与当前目标缺少可信关联。"))
        typed_field_id = str(before_input.states.get('input_field_id') or '').strip()
        if typed_field_id and typed_field_id != 'unknown':
            candidates = tuple((element for element in self._visible_elements(after, role='input') if
                (element.meaning == before_input.meaning == 'application_text_input')
                and (str(element.states.get('input_field_id') or '').strip() == typed_field_id)))
        else:
            candidates = self._matching_after_elements(before_input, after, overlap_fallback=True)
        reject_if(len(candidates) != 1, UniversalActionError("动作后无法唯一绑定原目标输入框。"))
        states = candidates[0].states
        reject_if('value' not in states or not isinstance(states['value'], str), UniversalActionError("动作后缺少输入框 states.value 精确文字证据。"))
        actual = states["value"]
        preedit_pending = False
        if resolved.input_method == 'chinese_pinyin':
            reject_if(actual != resolved.prior_input_value, UniversalActionError('拼音键入后应用输入值在候选确认前已意外变化。'))
            reject_if(states.get('ime_preedit_text') != resolved.input_pinyin, UniversalActionError("动作后缺少逐字一致的拼音组合证据。"))
            reject_if(states.get('ime_exact_candidate_text') != resolved.input_fragment, UniversalActionError("动作后缺少唯一逐字一致的中文候选。"))
            preedit_pending = True
        elif (
            resolved.kind == "input_verified_text"
            and resolved.input_method == "direct_latin"
            and actual != expected
        ):
            # Exact Latin preedit is progress only when one local candidate binds the same typed field and value.
            reject_if(not self._is_exact_direct_latin_preedit_transition(resolved, after, candidates[0]), UniversalActionError(f'动作后输入框文字不匹配：实际 {actual!r}，预期 {expected!r}。'))
            preedit_pending = True
        elif actual != expected:
            raise UniversalActionError(f'动作后输入框文字不匹配：实际 {actual!r}，预期 {expected!r}。')
        if resolved.kind == 'clear_verified_text' and before_input.states.get('ime_preedit_text'):
            reject_if(states.get('ime_preedit_text') not in (None, ''), UniversalActionError("清空动作后输入法预编辑文字仍未清除。"))
            reject_if(states.get('focused') is not True, UniversalActionError("清空动作后原typed输入框不再聚焦。"))
        after_input = candidates[0]
        reject_if(not self._input_scene_identity_is_stable(before, after, before_input, after_input), UniversalActionError("输入动作后 App 或页面身份发生变化。"))
        return preedit_pending

    def _is_exact_direct_latin_preedit_transition(self, resolved: ResolvedSemanticAction, after: UIScene,
        after_input: UIElement) -> bool:
        expected = resolved.expected_input_value
        fragment = resolved.input_fragment
        prior = resolved.prior_input_value
        states = after_input.states
        exact_candidates = tuple((element for element in after.elements if element.meaning == 'ime_exact_candidate'
            and element.label == fragment and (float(element.confidence) >= self.min_confidence)
            and (element.states.get('goal_relevant') is True) and (element.states.get('fully_visible') is True)
            and (element.states.get('ime_candidate') is True)
            and (element.states.get('input_element_id') == after_input.element_id)
            and (element.states.get('prior_input_value') == prior)
            and (element.states.get('expected_input_value') == expected)
            and (element.states.get('pinyin') == fragment)))
        return bool(resolved.kind == 'input_verified_text' and resolved.input_method == 'direct_latin'
            and isinstance(prior, str) and isinstance(fragment,
            str) and fragment and (expected == prior + fragment) and (states.get('value') == prior)
            and (states.get('ime_preedit_text') == fragment) and (states.get('ime_exact_candidate_text') == fragment)
            and (len(exact_candidates) == 1))

    @classmethod
    def _input_scene_identity_is_stable(cls, before: UIScene, after: UIScene, before_input: UIElement,
        after_input: UIElement) -> bool:
        if before.foreground_app_id == after.foreground_app_id and before.screen_id == after.screen_id:
            return True

        app_identity_compatible = input_app_identity_compatible(before.foreground_app_id, after.foreground_app_id)
        before_states = before_input.states
        after_states = after_input.states
        typed_field_id = str(before_states.get("input_field_id") or "").strip()
        before_field_label = str(before_states.get('input_field_label') or '').strip()
        after_field_label = str(after_states.get('input_field_label') or '').strip()
        typed_field_identity = bool(typed_field_id and typed_field_id != 'unknown'
            and (str(after_states.get('input_field_id') or '').strip() == typed_field_id)
            and (before_input.meaning == after_input.meaning == 'application_text_input')
            and (before_states.get('fully_visible') is not False) and (after_states.get('fully_visible') is not False)
            and isinstance(before_states.get('input_multiline'),
            bool) and (after_states.get('input_multiline') == before_states.get('input_multiline'))
            and (not (before_field_label and after_field_label and (before_field_label != after_field_label))))
        screen_identity_compatible = input_screen_identity_compatible(before.screen_id, after.screen_id)
        if not screen_identity_compatible and (not typed_field_identity):
            return False
        if not app_identity_compatible:
            if (input_app_identity_is_concrete_package(before.foreground_app_id)
                or input_app_identity_is_concrete_package(after.foreground_app_id)):
                return False
            before_family = input_screen_identity_family(before.screen_id)
            after_family = input_screen_identity_family(after.screen_id)
            if (not before_family or before_family != after_family) and (not typed_field_identity):
                return False
        if not (cls._regions_stably_overlap(before_input.bounds, after_input.bounds) or typed_field_identity):
            return False
        if before_states.get('focused') is not True or after_states.get('focused') is not True:
            return False
        before_layout = before_states.get("keyboard_layout")
        after_layout = after_states.get("keyboard_layout")
        if before_layout in {None, 'unknown'} or after_layout in {None, 'unknown'} or before_layout != after_layout:
            return False
        before_mode = before_states.get("keyboard_input_mode")
        after_mode = after_states.get("keyboard_input_mode")
        preedit_clear_transition = bool(typed_field_identity and before_states.get('value') == ''
            and isinstance(before_states.get('ime_preedit_text'),
            str) and bool(before_states.get('ime_preedit_text')) and (after_states.get('value') == '')
            and (after_states.get('ime_preedit_text') in {None, ''}))
        preedit_start_transition = bool(typed_field_identity and isinstance(before_states.get('value'),
            str) and (after_states.get('value') == before_states.get('value'))
            and (before_states.get('ime_preedit_text') in {None,
            ''}) and isinstance(after_states.get('ime_preedit_text'),
            str) and bool(after_states.get('ime_preedit_text'))
            and (after_states.get('ime_exact_candidate_text') == after_states.get('ime_preedit_text')))
        if (before_mode in {None, 'unknown'} or after_mode in {None,
            'unknown'} or (before_mode != after_mode and (not preedit_clear_transition)
            and (not preedit_start_transition))):
            return False
        return (
            before.camera_alignment.camera_layout_orientation
            == after.camera_alignment.camera_layout_orientation
            and before.camera_alignment.phone_content_rotation
            == after.camera_alignment.phone_content_rotation
        )

    @classmethod
    def scenes_semantically_equivalent(cls, before: UIScene, after: UIScene) -> bool:
        """Ignore camera noise and model box jitter when comparing scenes."""

        def freeze(value: Any) -> Any:
            if isinstance(value, dict):
                return tuple(sorted(((str(key), freeze(item)) for key, item in value.items())))
            if isinstance(value, (list, tuple)):
                return tuple(freeze(item) for item in value)
            return value

        def signature(scene: UIScene) -> tuple[Any, ...]:
            elements = tuple(
                sorted(
                    (
                        element.role.casefold(),
                        element.label.casefold(),
                        # Prefer visible text plus role/state; unlabeled icons still need meaning.
                        (
                            ""
                            if element.label.strip()
                            else element.meaning.casefold()
                        ),
                        freeze(element.states),
                    )
                    for element in scene.elements
                )
            )
            return (scene.foreground_app_id.casefold(), scene.screen_id.casefold(),
                tuple(sorted((item.casefold() for item in scene.overlays))), elements)

        before.validate()
        after.validate()
        return signature(before) == signature(after)

    def _resolve_target(self, action: SemanticAction, scene: UIScene, *, prefix: str='',
        required_role: str | None=None) -> UIElement:
        target = str(action.params.get(f"{prefix}target") or "").strip()
        element_id = str(action.params.get(f"{prefix}element_id") or "").strip()
        reject_if(not target and (not element_id), UniversalActionError(f'{action.action} 缺少 {prefix}target 或 {prefix}element_id。'))
        role = str(action.params.get(f"{prefix}role") or "").strip() or None
        if required_role is not None:
            reject_if(role is not None and role != required_role, UniversalActionError(f"动作目标角色必须为 {required_role}。"))
            role = required_role
        label = str(action.params.get(f"{prefix}label") or "").strip() or None
        states = action.params.get(f"{prefix}states") or {}
        reject_if(not isinstance(states, dict), UniversalActionError(f"{action.action}.{prefix}states 格式无效。"))
        try:
            if element_id:
                element = scene.get_element(element_id, min_confidence=self.min_confidence)
                reject_if(target and target.casefold() not in {element.meaning.casefold(), element.label.casefold()}, UISceneError(f'元素 {element_id} 的语义与目标不一致：{target}'))
                reject_if(role and element.role != role, UISceneError(f'元素 {element_id} 的角色与目标不一致：{role}'))
                reject_if(label and element.label.casefold() != label.casefold(), UISceneError(f'元素 {element_id} 的文字与目标不一致：{label}'))
                reject_if(any((element.states.get(key) != value for key, value in states.items())), UISceneError(f"元素 {element_id} 的状态与目标不一致。"))
                return element
            return scene.resolve_unique(meaning=target, label=label, role=role, states=states,
                min_confidence=self.min_confidence)
        except UISceneError as exc:
            raise UniversalActionError(str(exc)) from exc

    @staticmethod
    def _point_action(action: SemanticAction, element: UIElement, expected_effect: dict[str, Any],
        before_fingerprint: str, *, scene: UIScene | None=None,
        local_point_grounding: LocalPointGrounding | None=None) -> ResolvedSemanticAction:
        reject_if(element.role == 'container', UniversalActionError("页面容器不是可点击控件，禁止执行点击。"))
        normalized_point = element.center
        proposed_point: NormalizedPoint | None = None
        grounding_payload: dict[str, Any] | None = None
        if local_point_grounding is not None:
            reject_if(scene is None, UniversalActionError("本地落点证据缺少当前场景绑定。"))
            local_point_grounding.validate_for(scene, element)
            proposed_point = element.center
            normalized_point = local_point_grounding.grounded_point
            grounding_payload = local_point_grounding.to_dict()
            if action.action in {'double_tap', 'long_press'}:
                UniversalActionController._validate_gesture_point(normalized_point, label='本地修正手势落点')
        return ResolvedSemanticAction(node_id=action.node_id, kind=action.action, normalized_point=normalized_point,
            target_element_id=element.element_id, before_fingerprint=before_fingerprint,
            expected_effect=expected_effect, formal_candidate_id=str(action.params.get('formal_candidate_id')
            or '').strip(), formal_transition=dict(action.params.get('formal_transition') or {}),
            proposed_normalized_point=proposed_point, point_grounding=grounding_payload)
