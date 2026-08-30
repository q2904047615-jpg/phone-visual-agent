from __future__ import annotations

from .validation import canonical_digest, dataclass_wire, reject_if
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .task_semantic_ir import ConstraintIntent, EffectIntent, SemanticEntity, TaskSemanticIR
from .semantic_action import SemanticAction
from .canonical_action_kinds import CANONICAL_ACTION_KINDS, expected_idempotent_system_surface_kind
from .text_input_utils import normalize_user_text
from .text_transport import TEXT_TRANSPORT_MAX_FRAGMENT_UTF8_BYTES, TextTransportProfile
from .ui_scene import UIElement, UIScene, scene_matches_target_app_surface, scene_surface_kind
from .verified_text_transaction import VerifiedTextTransactionError, plan_from_input_states


CANONICAL_ACTION_PROTOCOL = "2026-08-20-canonical-action-v1"

_EFFECT_CONTROL_MEANINGS = {'send_message': frozenset({'send_message'}),
    'publish_content': frozenset({'publish_content', 'publish', 'post_content', 'comment', 'reply'}),
    'relationship_change': frozenset({'follow', 'unfollow', 'subscribe', 'unsubscribe', 'favorite', 'unfavorite'}),
    'membership_change': frozenset({'join', 'leave', 'invite', 'remove_member'})}
_ALL_EFFECT_CONTROL_MEANINGS = frozenset((meaning for meanings in _EFFECT_CONTROL_MEANINGS.values() for meaning
    in meanings))


def _element_realizes_effect(element: UIElement, effect_kind: str) -> bool:
    """Bind a typed effect only to its canonical visible action control."""

    meanings = _EFFECT_CONTROL_MEANINGS.get(effect_kind, frozenset())
    return bool(meanings and element.role in {'button', 'icon',
        'toggle'} and (element.meaning in meanings) and (element.states.get('visible') is not False)
        and (element.states.get('enabled') is not False) and (element.states.get('fully_visible') is True))

MIN_ELEMENT_CONFIDENCE = 0.72
MIN_READY_CANDIDATES = 1
MAX_READY_CANDIDATES = 24

ELEMENT_ACTION_ROLES = frozenset({'button', 'icon', 'input', 'tab', 'toggle', 'list_item'})
EXPECTATION_OPERATORS = frozenset({'equals', 'not_equals', 'present', 'absent', 'changed'})
EXPECTATION_PREDICATES = frozenset({'surface.kind', 'surface.active_ref', 'surface.focused_entity_ref',
    'surface.overlay_present', 'surface.navigation_depth', 'surface.viewport', 'system_ui.navigation_bar_visible',
    'observation.changed', 'scene.changed', 'element.exists', 'element.state.focused', 'element.state.value',
    'element.state.ime_preedit_text', 'element.state.ime_exact_candidate_text', 'element.state.keyboard_layout',
    'element.state.keyboard_input_mode', 'element.state.keyboard_case_mode', 'element.state.interaction_result',
    'element.state.location_relation', 'input_field.focused', 'effect.applied'})
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
class CanonicalActionProtocolError(ValueError):
    pass


_FORMAL_AUTHORITY_PARAMS = frozenset({'formal_candidate_id', 'formal_transition'})


@dataclass(frozen=True)
class GenericStepProposal:
    """Transport one catalog-selected action or local blocked state, never task completion."""

    status: str
    action: SemanticAction | None = None
    reason: str = ""

    def validate(self, scene: UIScene) -> None:
        scene.validate()
        reject_if(self.status not in {'action', 'blocked'}, CanonicalActionProtocolError(f"不支持的单步状态：{self.status}"))
        if self.status == 'blocked':
            reject_if(self.action is not None, CanonicalActionProtocolError("blocked 状态不能携带动作。"))
            reject_if(not self.reason.strip(), CanonicalActionProtocolError("阻塞报告必须说明原因。"))
            return
        reject_if(self.action is None, CanonicalActionProtocolError("action 状态缺少唯一动作。"))

        kind, params = self.action.action, self.action.params
        reject_if(kind not in CANONICAL_ACTION_KINDS, CanonicalActionProtocolError(f"单步动作不在 canonical 动作集合：{kind}"))
        element_actions = {'tap_semantic', 'dismiss_overlay', 'input_verified_text', 'press_enter',
            'clear_verified_text', 'double_tap', 'long_press'}
        element = None
        if kind in element_actions:
            element_id = str(params.get("element_id") or "").strip()
            reject_if(not element_id, CanonicalActionProtocolError("元素动作必须引用当前场景 element_id。"))
            element = scene.get_element(element_id)

        if kind == 'input_verified_text':
            text = params.get("text")
            reject_if(not isinstance(text, str) or not text or len(text) > 4000 or ('\r' in text), CanonicalActionProtocolError("输入动作 text 必须为1～4000字符；换行由可见 Enter 键分段执行。"))
            reject_if(element.role != 'input', CanonicalActionProtocolError("输入动作必须绑定 input 元素。"))
            if params.get('text_transport') == 'companion_ime':
                required = {'input_field_id', 'prior_input_value', 'input_fragment', 'expected_input_value'}
                reject_if(any(name not in params for name in required), CanonicalActionProtocolError("Companion IME 输入动作缺少 typed 文字事务字段。"))
                prior = params['prior_input_value']
                fragment = params['input_fragment']
                expected = params['expected_input_value']
                reject_if(not isinstance(prior, str) or not isinstance(fragment, str) or not fragment
                    or not isinstance(expected, str) or expected != prior + fragment or expected != text,
                    CanonicalActionProtocolError("Companion IME 输入动作的 prior/fragment/expected 不一致。"))
                reject_if(str(params['input_field_id'] or '').strip() in {'', 'unknown'},
                    CanonicalActionProtocolError("Companion IME 输入动作缺少 typed input_field_id。"))
        elif kind == 'clear_verified_text':
            allowed = {'element_id', 'target', 'role', 'label', 'states', 'expected_effect', 'text_transport',
                'input_field_id', 'prior_input_value', 'expected_input_value', *_FORMAL_AUTHORITY_PARAMS}
            unexpected = set(params) - allowed
            reject_if(unexpected, CanonicalActionProtocolError("清空动作包含协议外参数：" + ", ".join(sorted(unexpected))))
            reject_if(element.role != 'input', CanonicalActionProtocolError("清空动作必须绑定 input 元素。"))
            if params.get('text_transport') == 'companion_ime':
                reject_if(str(params.get('input_field_id') or '').strip() in {'', 'unknown'}
                    or not isinstance(params.get('prior_input_value'), str)
                    or params.get('expected_input_value') != '',
                    CanonicalActionProtocolError("Companion IME 清空动作缺少精确 typed 文字事务。"))
        elif kind == 'long_press':
            duration_ms = params.get("duration_ms", 800)
            reject_if(isinstance(duration_ms, bool) or not isinstance(duration_ms, (int, float)) or (not 500 <= float(duration_ms) <= 2000), CanonicalActionProtocolError("长按 duration_ms 必须在500～2000之间。"))
        elif kind == 'drag':
            source_id = str(params.get("source_element_id") or "").strip()
            destination_id = str(params.get("destination_element_id") or "").strip()
            reject_if(not source_id or not destination_id or source_id == destination_id, CanonicalActionProtocolError("拖动必须绑定两个不同的可信元素。"))
            scene.get_element(source_id)
            scene.get_element(destination_id)
        elif kind == 'swipe':
            reject_if(str(params.get('direction') or '').strip() not in {'up', 'down', 'left', 'right'}, CanonicalActionProtocolError("滑动动作方向无效。"))
        elif kind == 'reveal_system_navigation':
            unexpected = set(params) - {"expected_effect"} - _FORMAL_AUTHORITY_PARAMS
            reject_if(unexpected, CanonicalActionProtocolError("系统导航栏唤出动作不能携带坐标、方向、距离或其他参数。"))
            system_ui = scene.system_ui
            reject_if(system_ui.immersive_or_fullscreen is not True or system_ui.navigation_bar_visible is not False, CanonicalActionProtocolError("系统导航栏唤出动作要求当前画面明确处于沉浸态且导航栏隐藏。"))
            reject_if(params.get('expected_effect') != {'system_ui': {'navigation_bar_visible': True}}, CanonicalActionProtocolError("系统导航栏唤出动作必须精确声明结构化导航栏可见后置条件。"))
        elif kind == 'launch_app':
            allowed = {'target_surface_id', 'target_app_id', 'target_app_name', 'launch_ref', 'expected_app_id',
                'expected_effect', *_FORMAL_AUTHORITY_PARAMS}
            reject_if(set(params) != allowed, CanonicalActionProtocolError("App 直启动作字段不完整或包含协议外参数。"))
            for key in ('target_surface_id', 'target_app_id', 'target_app_name', 'launch_ref', 'expected_app_id'):
                reject_if(not isinstance(params.get(key), str) or not params[key].strip(),
                    CanonicalActionProtocolError(f"App 直启动作缺少 {key}。"))
            reject_if(params['expected_effect'] != {'app_id': params['target_app_id']},
                CanonicalActionProtocolError("App 直启必须绑定 typed 目标 App 的视觉后置条件。"))

    def to_dict(self) -> dict[str, Any]:
        return _record_wire(self, "proposal")


def reject_raw_control_data(value: Any) -> None:
    """Reject model-authored coordinates or multi-action payloads."""

    forbidden = {'actions', 'steps', 'plan', 'coordinate', 'coordinates', 'tap_point', 'x', 'y', 'shell', 'command'}
    if isinstance(value, dict):
        for (key, item) in value.items():
            reject_if(str(key).strip().lower() in forbidden, CanonicalActionProtocolError(f'单步动作包含禁止字段：{key}'))
            reject_raw_control_data(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_raw_control_data(item)
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise CanonicalActionProtocolError("单步动作参数类型无效。")


def _json_value(value: Any, field_name: str) -> Any:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise CanonicalActionProtocolError(f"{field_name} 必须是可序列化 JSON 值。") from exc


def _record_wire(value: Any, field_name: str, *, omit: Iterable[str]=()) -> dict[str, Any]:
    result = _json_value(dataclass_wire(value), field_name)
    for key in omit:
        result.pop(key, None)
    return result


class _ValidatedRecordWire:
    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _record_wire(self, type(self).__name__)


def _digest(value: Any) -> str:
    return canonical_digest(value)


def _stable_id(prefix: str, value: Any) -> str:
    return f"{prefix}_{_digest(value)[:20]}"


def _element_ref(element_id: str) -> str:
    return _stable_id("element", {"source_element_id": str(element_id)})


def _validate_id(value: str, field_name: str) -> None:
    reject_if(not isinstance(value, str) or not _ID_PATTERN.fullmatch(value), CanonicalActionProtocolError(f"{field_name} 无效：{value!r}"))


def _required_text(value: Any, field_name: str, *, max_length: int=300) -> str:
    reject_if(not isinstance(value, str) or not value.strip(), CanonicalActionProtocolError(f"{field_name} 必须是非空字符串。"))
    text = value.strip()
    reject_if(len(text) > max_length, CanonicalActionProtocolError(f"{field_name} 超过长度限制。"))
    return text


def _surface_kind(scene: UIScene) -> str:
    return scene_surface_kind(scene)


@dataclass(frozen=True)
class StateExpectation:
    subject_ref: str
    predicate: str
    operator: str
    value: Any = None

    def validate(self) -> None:
        _validate_id(self.subject_ref, "expectation.subject_ref")
        _required_text(self.predicate, "expectation.predicate", max_length=100)
        reject_if(self.predicate not in EXPECTATION_PREDICATES, CanonicalActionProtocolError(f'expectation.predicate 无效：{self.predicate}'))
        reject_if(self.operator not in EXPECTATION_OPERATORS, CanonicalActionProtocolError(f'expectation.operator 无效：{self.operator}'))
        if self.operator in {'equals', 'not_equals'}:
            _json_value(self.value, "expectation.value")
        elif self.value is not None:
            raise CanonicalActionProtocolError(f'expectation.{self.operator} 不得携带 value。')

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        omit = () if self.operator in {"equals", "not_equals"} else ("value",)
        return _record_wire(self, "expectation", omit=omit)

    @classmethod
    def from_dict(cls, value: Any) -> 'StateExpectation':
        required = {"subject_ref", "predicate", "operator"}
        reject_if(not isinstance(value, Mapping) or not required.issubset(value)
            or set(value) - (required | {"value"}), CanonicalActionProtocolError("expectation wire 结构无效。"))
        operator = str(value.get("operator") or "")
        if operator in {'equals', 'not_equals'}:
            reject_if('value' not in value, CanonicalActionProtocolError("expectation 等值条件缺少 value。"))
        else:
            reject_if(value.get('value') is not None,
                CanonicalActionProtocolError("expectation 非等值条件不得携带 value。"))
        expectation = cls(subject_ref=str(value.get('subject_ref') or ''),
            predicate=str(value.get('predicate') or ''), operator=operator,
            value=value.get('value') if operator in {'equals', 'not_equals'} else None)
        expectation.validate()
        return expectation


@dataclass(frozen=True)
class TypedStateTransition(_ValidatedRecordWire):
    transition_id: str
    expectations: tuple[StateExpectation, ...]
    exploratory: bool = False

    def validate(self) -> None:
        _validate_id(self.transition_id, "transition.transition_id")
        reject_if(not self.expectations, CanonicalActionProtocolError("transition.expectations 不能为空。"))
        for expectation in self.expectations:
            expectation.validate()
            reject_if(expectation.predicate in {'scene.changed', 'observation.changed'} and (not self.exploratory), CanonicalActionProtocolError('scene/observation changed 只能用于 exploratory transition。'))
        reject_if(not isinstance(self.exploratory, bool), CanonicalActionProtocolError("transition.exploratory 必须是布尔值。"))

@dataclass(frozen=True)
class CanonicalActionCandidate(_ValidatedRecordWire):
    candidate_id: str
    action_kind: str
    transition: TypedStateTransition
    parameters: dict[str, Any] = field(default_factory=dict)
    effect_ref: str = ""

    def validate(self) -> None:
        _validate_id(self.candidate_id, "candidate.candidate_id")
        reject_if(self.action_kind not in CANONICAL_ACTION_KINDS, CanonicalActionProtocolError(f"candidate.action_kind 无效：{self.action_kind}"))
        _json_value(self.parameters, "candidate.parameters")
        reject_if(any((key in self.parameters for key in {'bounds', 'point', 'x', 'y'})), CanonicalActionProtocolError("canonical candidate 不得携带坐标。"))
        if self.effect_ref:
            _validate_id(self.effect_ref, "candidate.effect_ref")
        self.transition.validate()

@dataclass(frozen=True)
class CanonicalActionCatalog(_ValidatedRecordWire):
    task_id: str
    device_id: str
    revision: int
    candidates: tuple[CanonicalActionCandidate, ...]
    status: str
    warnings: tuple[str, ...] = ()
    protocol_version: str = CANONICAL_ACTION_PROTOCOL

    def validate(self) -> None:
        reject_if(self.protocol_version != CANONICAL_ACTION_PROTOCOL, CanonicalActionProtocolError("canonical action protocol_version 无效。"))
        _required_text(self.task_id, "report.task_id", max_length=128)
        _required_text(self.device_id, "report.device_id", max_length=128)
        reject_if(isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 1, CanonicalActionProtocolError("report.revision 必须是正整数。"))
        reject_if(self.status not in {'ready', 'blocked'}, CanonicalActionProtocolError("report.status 无效。"))
        reject_if(self.status == 'ready' and (not MIN_READY_CANDIDATES <= len(self.candidates) <= MAX_READY_CANDIDATES), CanonicalActionProtocolError(f'ready report 必须包含{MIN_READY_CANDIDATES}至{MAX_READY_CANDIDATES}个候选。'))
        reject_if(self.status == 'blocked' and len(self.candidates) >= MIN_READY_CANDIDATES, CanonicalActionProtocolError("候选已足够时不得标记 blocked。"))

        seen: set[str] = set()
        for candidate in self.candidates:
            candidate.validate()
            reject_if(candidate.candidate_id in seen, CanonicalActionProtocolError("report.candidate ID 重复。"))
            seen.add(candidate.candidate_id)

def _element_eligible(element: UIElement) -> bool:
    return element.role in ELEMENT_ACTION_ROLES and float(element.confidence) >= MIN_ELEMENT_CONFIDENCE and (
        element.states.get('enabled') is not False) and (element.states.get('visible')
        is not False) and (element.states.get('fully_visible') is True)


def _verified_text_affordance_ready(element: UIElement, target_text: str) -> bool:
    """Expose batch text input only when the next typed segment is executable."""

    if element.role != 'input' or element.states.get('focused') is not True:
        return False
    try:
        step = plan_from_input_states(target_text, element.states)
    except (ValueError, VerifiedTextTransactionError):
        return False
    if step is None or step.kind == 'literal_key':
        return False
    return bool(element.states.get('keyboard_layout') == 'qwerty'
        and element.states.get('keyboard_input_mode') == step.required_mode and (not step.required_case_mode
        or element.states.get('keyboard_case_mode') == step.required_case_mode)
        and (not element.states.get('ime_preedit_text')))


def _companion_text_step(element: UIElement, target_text: str) -> tuple[str, str, str] | None:
    """Return the sole Unicode append authorized by one focused typed field."""

    if element.role != 'input' or element.states.get('focused') is not True:
        return None
    input_field_id = str(element.states.get('input_field_id') or '').strip()
    current = element.states.get('value')
    if input_field_id in {'', 'unknown'} or not isinstance(current, str) or element.states.get('ime_preedit_text'):
        return None
    try:
        target = normalize_user_text(target_text, field_name='输入文字')
    except ValueError:
        return None
    if current == target or not target.startswith(current):
        return None
    remainder = target[len(current):]
    used = 0
    end = 0
    for char in remainder:
        size = len(char.encode('utf-8'))
        if used + size > TEXT_TRANSPORT_MAX_FRAGMENT_UTF8_BYTES:
            break
        used += size
        end += 1
    fragment = remainder[:end]
    if not fragment:
        return None
    return current, fragment, current + fragment


def _element_proves_scrollable_viewport(element: UIElement) -> bool:
    """Grant swipe affordance only from a typed, evidenced viewport fact."""

    base_evidence = bool(element.role == 'container' and float(element.confidence) >= MIN_ELEMENT_CONFIDENCE
        and (element.states.get('visible') is not False) and (element.states.get('scrollable') is True)
        and (element.states.get('scroll_axis') in {'vertical',
        'horizontal'}) and any((str(item).strip() for item in element.evidence)))
    if not base_evidence:
        return False
    if element.states.get("fully_visible") is True:
        # Preserve the already verified cross-App swipe contract.
        return True
    # For a scrollable viewport, fully_visible=false is the typed content-crop
    # fact; non-empty element evidence above must independently support it.
    return bool(element.states.get('fully_visible') is False and element.states.get('goal_relevant') is True)


def _swipe_directions_for_viewport(element: UIElement) -> tuple[str, ...]:
    """Return only directions supported by the observed viewport axis/edge."""

    axis = element.states.get("scroll_axis")
    if axis == 'horizontal':
        backward, forward = "right", "left"
    elif axis == 'vertical':
        backward, forward = "down", "up"
    else:
        return ()
    page_index = element.states.get("page_index")
    page_count = element.states.get("page_count")
    if (isinstance(page_index, int) and (not isinstance(page_index, bool)) and isinstance(page_count,
        int) and (not isinstance(page_count, bool)) and (page_count >= 2) and (0 <= page_index < page_count)):
        directions: list[str] = []
        if page_index < page_count - 1:
            directions.append(forward)
        if page_index > 0:
            directions.append(backward)
        return tuple(directions)
    return (forward, backward)


def _swipe_direction_patterns(chinese: str, english: str) -> tuple[re.Pattern[str], ...]:
    long_form = rf"{english}ward(?:s)?"
    action = r"(?:swipe|flick|drag)"
    return (
        re.compile(rf"(?:向|往|朝)\s*{chinese}(?!方)"),
        re.compile(rf"{chinese}\s*(?:滑|划|拖|推)"),
        re.compile(rf"\b{action}\s+{english}\b", re.IGNORECASE),
        re.compile(rf"\b{action}\b[^.;!?\n]{{0,80}}\b{long_form}\b", re.IGNORECASE),
        re.compile(
            rf"\b{action}\b[^.;!?\n]{{0,80}}\b{english}\b"
            r"\s*(?:away|off|out)?\s*(?:[.;!?]|$)",
            re.IGNORECASE,
        ),
        re.compile(rf"\b{long_form}\s+{action}\b", re.IGNORECASE),
    )


_EXPLICIT_SWIPE_DIRECTION_PATTERNS = {direction: _swipe_direction_patterns(chinese, direction) for direction,
    chinese in {'up': '上', 'down': '下', 'left': '左', 'right': '右'}.items()}


def _explicit_required_swipe_direction(constraints: Iterable[ConstraintIntent]) -> str | None:
    """Narrow a typed swipe with one explicit direction; never mint a swipe from prose alone."""

    source_text = ' '.join((item.source_text for item in constraints if item.kind == 'required_action'
        and item.value == 'swipe' and item.authoritative and item.source_text.strip()))
    matches = {direction for direction, patterns in _EXPLICIT_SWIPE_DIRECTION_PATTERNS.items() if any((pattern.search(
        source_text) for pattern in patterns))}
    if len(matches) != 1:
        return None
    return next(iter(matches))


def _unique_directional_swipe_presence(scene: UIScene) -> UIElement | None:
    """Anchor a typed directional swipe to one observed object without emitting coordinates."""

    target = scene.unique_trusted_goal_element()
    if target is None:
        completion_evidence = scene.trusted_completion_evidence()
        if len(completion_evidence) != 1:
            return None
        target = completion_evidence[0]
        if (any((other.element_id != target.element_id and other.states.get('goal_relevant') is True
            and (float(other.confidence) >= MIN_ELEMENT_CONFIDENCE) for other in scene.elements))):
            return None
    if target.states.get('fully_visible') is not True:
        return None
    if not any((str(item).strip() for item in target.evidence)):
        return None
    return target


def _exact_tap_text_target_eligible(element: UIElement) -> bool:
    """Admit only an exact-authorized visible text target for a semantic tap."""

    return element.role == 'text' and float(element.confidence) >= MIN_ELEMENT_CONFIDENCE and (element.states.get(
        'enabled') is not False) and (element.states.get('visible')
        is not False) and (element.states.get('fully_visible') is True) and (element.states.get('goal_relevant')
        is True)


def _unique_exact_matches(elements: tuple[UIElement, ...], entity: SemanticEntity, *,
    allow_exact_tap_text: bool=False) -> tuple[UIElement, ...]:
    if not isinstance(entity.value, str) or not entity.value:
        return ()
    literal = entity.value.casefold()
    return tuple((element for element in elements if (_element_eligible(element) or (allow_exact_tap_text
        and _exact_tap_text_target_eligible(element))) and literal in {element.label.strip().casefold(),
        str(element.states.get('value', '')).strip().casefold()}))


def _candidate(*, action_kind: str, expectations: tuple[StateExpectation, ...], exploratory: bool=False,
    parameters: Mapping[str, Any] | None=None, effect_ref: str='') -> CanonicalActionCandidate:
    transition_payload = {'action_kind': action_kind, 'expectations': [item.to_dict() for item in expectations],
        'exploratory': exploratory}
    transition = TypedStateTransition(transition_id=_stable_id('transition', transition_payload),
        expectations=expectations, exploratory=exploratory)
    payload = {'action_kind': action_kind, 'parameters': dict(parameters or {}), 'effect_ref': effect_ref,
        'transition': transition.to_dict()}
    return CanonicalActionCandidate(candidate_id=_stable_id('candidate', payload), action_kind=action_kind,
        transition=transition, parameters=dict(parameters or {}), effect_ref=effect_ref)


def compile_canonical_action_catalog(scene: UIScene, semantic_ir: TaskSemanticIR,
    available_action_kinds: Iterable[str], *, launch_target: Mapping[str, str] | None=None,
    text_transport_profile: TextTransportProfile | None=None
    ) -> CanonicalActionCatalog:
    """Compile the sole deterministic action catalog for the active subgoal."""

    scene.validate()
    semantic_ir.validate()
    active_subgoals = tuple((item for item in semantic_ir.subgoals if item.status == 'active'))
    reject_if(len(active_subgoals) != 1, CanonicalActionProtocolError('canonical action catalog 要求且只允许一个 active subgoal。'))
    active_subgoal = active_subgoals[0]
    constraints_by_id = {item.constraint_id: item for item in semantic_ir.constraints}
    active_required_action_constraints = tuple((constraints_by_id[ref] for ref in active_subgoal.constraint_refs if ref
        in constraints_by_id and constraints_by_id[ref].kind == 'required_action'))
    active_required_actions = frozenset((str(item.value) for item in active_required_action_constraints))
    active_input_fields = tuple((item for item in semantic_ir.input_fields if active_subgoal.subgoal_id
        in item.source_subgoal_ids))
    active_input_payload_refs = frozenset((item.payload_ref for item in active_input_fields))
    predecessor_input_field_ids = {item.field_id for item
        in semantic_ir.input_fields if set(item.source_subgoal_ids).intersection(active_subgoal.depends_on)}
    direct_successor_ids = {item.subgoal_id for item in semantic_ir.subgoals if active_subgoal.subgoal_id
        in item.depends_on}
    successor_input_field_ids = {item.field_id for item
        in semantic_ir.input_fields if set(item.source_subgoal_ids).intersection(direct_successor_ids)}
    if predecessor_input_field_ids and successor_input_field_ids:
        boundary_input_field_ids = predecessor_input_field_ids & successor_input_field_ids
    else:
        boundary_input_field_ids = predecessor_input_field_ids | successor_input_field_ids
    enter_input_field_ids = frozenset({item.field_id for item in active_input_fields} | boundary_input_field_ids)
    active_effect_refs = frozenset(active_subgoal.effect_refs)
    effects_by_id = {item.effect_id: item for item in semantic_ir.effects}
    active_entity_refs = set(active_subgoal.entity_refs)
    for effect_ref in active_effect_refs:
        effect = effects_by_id.get(effect_ref)
        if effect is not None:
            active_entity_refs.update(effect.target_refs)
            active_entity_refs.update(effect.payload_refs)
    active_entity_refs.update(active_input_payload_refs)
    desired_by_id = {item.state_id: item for item in semantic_ir.desired_states}
    active_desired_states = tuple((desired_by_id[ref] for ref in active_subgoal.desired_state_refs if ref
        in desired_by_id))
    active_entity_refs.update((item.subject_ref for item in active_desired_states if item.subject_ref
        in {entity.entity_id for entity in semantic_ir.entities}))
    active_text = ' '.join((str(item.value or '') for item in active_desired_states)).casefold()
    active_targets_input = bool(active_input_fields or 'clear_verified_text' in active_required_actions or any((token
        in active_text for token in ('input', 'text field', '输入框', '文本框', '编辑框'))))
    available = frozenset(str(value) for value in available_action_kinds)
    unknown = available - CANONICAL_ACTION_KINDS
    reject_if(unknown, CanonicalActionProtocolError('available_action_kinds 含未知动作：' + ', '.join(sorted(unknown))))
    if launch_target is not None:
        reject_if(not isinstance(launch_target, Mapping) or set(launch_target) != {'launch_ref', 'expected_app_id'}
            or any(not isinstance(launch_target.get(key), str) or not launch_target[key].strip()
            for key in ('launch_ref', 'expected_app_id')),
            CanonicalActionProtocolError("App 直启能力映射格式无效。"))
    companion_text = text_transport_profile is not None
    companion_append = False
    companion_clear = False
    if text_transport_profile is not None:
        try:
            text_transport_profile.validate()
        except ValueError as exc:
            raise CanonicalActionProtocolError(f"Companion IME profile 无效：{exc}") from exc
        reject_if(text_transport_profile.device_id != semantic_ir.device_id,
            CanonicalActionProtocolError("Companion IME profile 与当前 device_id 不一致。"))
        companion_text = bool(text_transport_profile.enabled)
        companion_append = companion_text and 'append_text' in text_transport_profile.capabilities
        companion_clear = companion_text and 'clear_text' in text_transport_profile.capabilities

    surface_ref = "surface_current"
    sorted_elements = tuple(sorted(scene.elements, key=lambda item: item.element_id))
    entity_by_id = {item.entity_id: item for item in semantic_ir.entities}
    active_input_payload_entities = tuple((entity_by_id[ref] for ref in sorted(active_input_payload_refs) if ref
        in entity_by_id and entity_by_id[ref].role == 'input_text' and isinstance(entity_by_id[ref].value, str)))
    effect_by_entity: dict[str, list[tuple[EffectIntent, str]]] = {}
    for effect in semantic_ir.effects:
        for entity_ref in effect.target_refs:
            effect_by_entity.setdefault(entity_ref, []).append((effect, "binds_effect_target"))
        for entity_ref in effect.payload_refs:
            effect_by_entity.setdefault(entity_ref, []).append((effect, "binds_effect_payload"))

    exact_elements_by_entity: dict[str, tuple[UIElement, ...]] = {}
    bindings_by_element: dict[str, set[tuple[str, str]]] = {item.element_id: set() for item in sorted_elements}
    relation_effects_by_element: dict[str, list[tuple[str, str, str]]] = {}

    def bind_element(element: UIElement, relation_kind: str, object_ref: str, *, entity_id: str='') -> None:
        bindings_by_element[element.element_id].add((relation_kind, object_ref))
        if entity_id:
            relation_effects_by_element.setdefault(element.element_id, []).append((object_ref, entity_id,
                relation_kind))

    focused_inputs = tuple((element for element in sorted_elements if _element_eligible(element)
        and element.role == 'input' and (element.states.get('focused') is True)))
    input_payload_entities = tuple((entity for entity in semantic_ir.entities if entity.role == 'input_text'
        and isinstance(entity.value, str)))
    if len(focused_inputs) == 1 and len(input_payload_entities) == 1:
        element = focused_inputs[0]
        entity = input_payload_entities[0]
        for (effect, relation_kind) in effect_by_entity.get(entity.entity_id, ()):
            if relation_kind != 'binds_effect_payload':
                continue
            bind_element(element, relation_kind, effect.effect_id, entity_id=entity.entity_id)
    if len(active_input_fields) == 1 and len(predecessor_input_field_ids) == 1:
        active_field = active_input_fields[0]
        predecessor_field_id = next(iter(predecessor_input_field_ids))
        predecessor_field = next((item for item in semantic_ir.input_fields if item.field_id == predecessor_field_id))
        predecessor_payload = entity_by_id.get(predecessor_field.payload_ref)
        focused_predecessors = [item for item in focused_inputs if item.states.get('input_field_id') ==
            predecessor_field_id and item.states.get('input_field_label') == predecessor_field.field_label
            and (predecessor_payload is not None) and (predecessor_payload.role == 'input_text')
            and (item.states.get('value') == predecessor_payload.value)]
        next_keys = tuple((element for element in sorted_elements if len(focused_predecessors) == 1
            and _element_eligible(element) and (element.meaning == 'input_next_field_key')
            and (element.role == 'button') and (element.states.get('input_next_field_key') is True)
            and (element.states.get('key_action') == 'next')
            and (element.states.get('source_input_field_id') == predecessor_field_id)
            and (element.states.get('target_input_field_id') == active_field.field_id)
            and (element.states.get('target_input_field_label') == active_field.field_label)))
        if len(next_keys) == 1:
            element = next_keys[0]
            bind_element(element, 'binds_next_input_field', active_field.field_id)
    exact_tap_authority = bool(active_subgoal.subgoal_id == 'exact_tap_semantic'
        and active_required_actions == {'tap_semantic'})
    exact_tap_text_element_ids: set[str] = set()
    for entity in semantic_ir.entities:
        allow_exact_tap_text = bool(exact_tap_authority and entity.role == 'target_ui_label')
        matches = _unique_exact_matches(sorted_elements, entity, allow_exact_tap_text=allow_exact_tap_text)
        exact_elements_by_entity[entity.entity_id] = matches
        if allow_exact_tap_text and len(matches) == 1 and (matches[0].role == 'text'):
            exact_tap_text_element_ids.add(matches[0].element_id)
        if len(matches) != 1:
            continue
        element = matches[0]
        bind_element(element, "exact_literal_match", entity.entity_id)
        for (effect, relation_kind) in effect_by_entity.get(entity.entity_id, ()):
            bind_element(element, relation_kind, effect.effect_id, entity_id=entity.entity_id)

    for surface in semantic_ir.surfaces:
        if surface.kind != 'app' or not surface.app_name:
            continue
        matches = tuple((element for element in sorted_elements if _element_eligible(element)
            and element.label.strip().casefold() == surface.app_name.casefold()))
        if len(matches) == 1:
            bind_element(matches[0], "binds_surface", surface.surface_id)

    scrollable_viewports = tuple((element for element in sorted_elements if _element_proves_scrollable_viewport(
        element)))
    target_swipe_presence: UIElement | None = None
    explicit_target_swipe_direction = _explicit_required_swipe_direction(active_required_action_constraints)
    if (not scrollable_viewports and active_required_actions == {'swipe'} and (explicit_target_swipe_direction
        is not None)):
        target_swipe_presence = _unique_directional_swipe_presence(scene)
    if len(scrollable_viewports) == 1:
        swipe_directions = _swipe_directions_for_viewport(scrollable_viewports[0])
    elif target_swipe_presence is not None:
        swipe_directions = (explicit_target_swipe_direction,)
    else:
        swipe_directions = ()
    def matches_active_input_field(element: UIElement, *, ambiguous: bool=False) -> bool:
        if len(active_input_fields) != 1:
            return ambiguous and active_targets_input
        field = active_input_fields[0]
        return bool(element.states.get('input_field_id') == field.field_id and (not field.field_label
            or element.states.get('input_field_label') == field.field_label) or (len(semantic_ir.input_fields) == 1
            and (not field.field_label)))

    system_actions = {'back', 'home', 'open_recent_apps', 'reveal_system_navigation', 'swipe', 'wait_for_change'}
    supported_system_actions = set(available & system_actions)
    if not swipe_directions:
        supported_system_actions.discard('swipe')
    if not (scene.system_ui.immersive_or_fullscreen is True
        and scene.system_ui.navigation_bar_visible is False):
        supported_system_actions.discard('reveal_system_navigation')
    supported_by_element: dict[str, set[str]] = {}
    for element in sorted_elements:
        normally_actionable = _element_eligible(element)
        exact_tap_text_target = element.element_id in exact_tap_text_element_ids
        if not normally_actionable and (not exact_tap_text_target):
            continue
        focus_only_input_surface = bool(element.role == 'input' and element.states.get('focus_only_input_surface')
            is True)
        input_auxiliary = element.meaning in {'ime_exact_candidate', 'input_exact_literal_key',
            'input_exact_enter_key', 'switch_keyboard_layout', 'switch_keyboard_case',
            'switch_keyboard_input_mode'}
        supported = set() if companion_text and input_auxiliary else set(available & {"tap_semantic"})
        if normally_actionable and (not focus_only_input_surface):
            supported.update(available & {"double_tap", "long_press", "drag"})
        if element.role == 'input' and element.states.get('focused') is True:
            companion_step = (_companion_text_step(element, active_input_payload_entities[0].value)
                if companion_append and len(active_input_payload_entities) == 1 else None)
            input_ready = companion_step is not None if companion_text else bool(len(active_input_payload_entities) == 1
                and _verified_text_affordance_ready(element, active_input_payload_entities[0].value))
            if ('input_verified_text' in available and len(active_input_payload_entities) == 1
                and matches_active_input_field(element) and input_ready):
                supported.add("input_verified_text")
            if ('clear_verified_text' in available and (not companion_text or companion_clear)
                and (bool(element.states.get('value'))
                or bool(element.states.get('ime_preedit_text')))):
                supported.add("clear_verified_text")
        if ('press_enter' in available and element.meaning == 'input_exact_enter_key'
            and (element.states.get('input_enter_key') is True) and (element.states.get('key_action') == 'newline')):
            supported.add("press_enter")
        if 'dismiss_overlay' in available and scene.overlays and (element.role in {'button', 'icon'}):
            supported.add("dismiss_overlay")
        supported_by_element[element.element_id] = supported

    candidates: list[CanonicalActionCandidate] = []
    active_external_effect_refs = tuple(active_subgoal.effect_refs if active_subgoal.external_impact ==
        'external_state' else ())
    unique_effect_control_by_ref: dict[str, str] = {}
    for effect_ref in active_external_effect_refs:
        effect = effects_by_id.get(effect_ref)
        if effect is None:
            continue
        matches = [element.element_id for element in sorted_elements if _element_eligible(element)
            and _element_realizes_effect(element, effect.kind)]
        if len(matches) == 1:
            unique_effect_control_by_ref[effect_ref] = matches[0]

    def append_element_candidate(element: UIElement, action_kind: str, expectations: Iterable[StateExpectation], *,
        exploratory: bool=False, effect_ref: str='') -> bool:
        if action_kind not in supported_by_element.get(element.element_id, set()):
            return False
        candidates.append(_candidate(action_kind=action_kind, expectations=tuple(expectations),
            exploratory=exploratory, parameters={'element_id': element.element_id}, effect_ref=effect_ref))
        return True

    element_by_id = {item.element_id: item for item in sorted_elements}

    def tap_transition(element: UIElement,
        bindings: set[tuple[str, str]]) -> tuple[tuple[StateExpectation, ...], str, bool] | None:
        element_ref = _element_ref(element.element_id)
        surface_binding = next((value for kind, value in bindings if kind == 'binds_surface'), None)
        extra: tuple[StateExpectation, ...] = ()
        if surface_binding is not None:
            expectation = StateExpectation(surface_ref, 'surface.active_ref', 'equals', surface_binding)
        elif element.meaning == 'input_next_field_key':
            field_id = str(element.states.get("target_input_field_id") or "").strip()
            if ('binds_next_input_field', field_id) not in bindings:
                return None
            expectation = StateExpectation(field_id, 'input_field.focused', 'equals', True)
        elif (element.meaning in {'ime_exact_candidate', 'input_exact_literal_key', 'input_exact_enter_key',
            'switch_keyboard_layout', 'switch_keyboard_case', 'switch_keyboard_input_mode'}):
            direct_value = element.meaning in {'ime_exact_candidate', 'input_exact_literal_key',
                'input_exact_enter_key'}
            value = element.states.get('expected_input_value' if direct_value else 'prior_input_value')
            if not isinstance(value, str):
                return None
            subject = _element_ref(element_by_id.get(str(element.states.get('input_element_id')
                or '')).element_id) if str(element.states.get('input_element_id')
                or '') in element_by_id else element_ref
            expectation = StateExpectation(subject, 'element.state.value', 'equals', value)
            switch_specs = {'switch_keyboard_layout': ('element.state.keyboard_layout', 'target_layout'),
                'switch_keyboard_case': ('element.state.keyboard_case_mode', 'target_mode'),
                'switch_keyboard_input_mode': ('element.state.keyboard_input_mode', 'target_mode')}
            if element.meaning in switch_specs:
                predicate, state_key = switch_specs[element.meaning]
                target = element.states.get(state_key)
                if isinstance(target, str) and target:
                    extra = (StateExpectation(subject, predicate, "equals", target),)
        elif element.role == 'input':
            expectation = StateExpectation(element_ref, 'element.state.focused', 'equals', True)
        else:
            entity_ref = next(iter(sorted(value for kind, value in bindings if kind == 'exact_literal_match')), "")
            expectation = StateExpectation(surface_ref, 'surface.focused_entity_ref', 'equals',
                entity_ref) if entity_ref else StateExpectation(surface_ref, 'surface.navigation_depth', 'changed')
        effect_ref = next((ref for ref, element_id in unique_effect_control_by_ref.items() if element_id ==
            element.element_id), '')
        if effect_ref:
            expectation = StateExpectation(effect_ref, 'effect.applied', 'equals', True)
        exploratory = bool(not effect_ref and surface_binding is None and element.role != 'input'
            and (element.meaning not in {'ime_exact_candidate', 'input_exact_literal_key', 'input_exact_enter_key',
            'switch_keyboard_layout', 'switch_keyboard_case', 'switch_keyboard_input_mode', 'input_next_field_key'}))
        return (expectation, *extra), effect_ref, exploratory

    for element in sorted_elements:
        element_ref = _element_ref(element.element_id)
        bindings = bindings_by_element[element.element_id]

        if ('tap_semantic' in supported_by_element.get(element.element_id, set())
            and not (element.role == 'input' and element.states.get('focused') is True)):
            tap_spec = tap_transition(element, bindings)
            if tap_spec is None:
                continue
            expectations, effect_ref, exploratory = tap_spec
            append_element_candidate(element, 'tap_semantic', expectations, exploratory=exploratory,
                effect_ref=effect_ref)

        if 'press_enter' in supported_by_element.get(element.element_id, set()):
            expected_value = element.states.get("expected_input_value")
            input_element_id = str(element.states.get("input_element_id") or "").strip()
            input_element = next((item for item in sorted_elements if item.element_id == input_element_id
                and item.role == 'input'), None)
            if isinstance(expected_value, str) and input_element is not None:
                append_element_candidate(element, 'press_enter',
                    (StateExpectation(_element_ref(input_element.element_id), 'element.state.value', 'equals',
                    expected_value),))

        if 'input_verified_text' in supported_by_element.get(element.element_id, set()):
            payload_entities = [entity_by_id[entity_id] for _, entity_id,
                relation_kind in relation_effects_by_element.get(element.element_id,
                ()) if relation_kind == 'binds_effect_payload' and entity_id in active_input_payload_refs
                and (entity_id in entity_by_id) and (entity_by_id[entity_id].role == 'input_text')]
            # A focused empty input does not literally contain the future payload.
            # Bind the unique typed input_text payload directly to the input affordance.
            if not payload_entities:
                payload_entities = list(active_input_payload_entities)
            if len(payload_entities) == 1:
                payload = payload_entities[0]
                if companion_text:
                    companion_step = _companion_text_step(element, payload.value)
                    if companion_step is None:
                        continue
                    prior, fragment, expected = companion_step
                    candidates.append(_candidate(action_kind='input_verified_text', expectations=(StateExpectation(
                        element_ref, 'element.state.value', 'equals', expected),), parameters={
                        'element_id': element.element_id, 'text_transport': 'companion_ime',
                        'input_field_id': str(element.states.get('input_field_id') or ''),
                        'prior_input_value': prior, 'input_fragment': fragment,
                        'expected_input_value': expected}))
                    continue
                try:
                    deterministic_input_step = plan_from_input_states(payload.value, element.states)
                except (ValueError, VerifiedTextTransactionError):
                    deterministic_input_step = None
                if deterministic_input_step is None:
                    continue
                expected_input_states = (
                    {
                        "value": deterministic_input_step.current_text,
                        "ime_preedit_text": deterministic_input_step.pinyin,
                        "ime_exact_candidate_text": deterministic_input_step.segment,
                    }
                    if deterministic_input_step.kind == "chinese_pinyin"
                    else {"value": deterministic_input_step.expected_value}
                )
                candidates.append(_candidate(action_kind='input_verified_text',
                    expectations=tuple((StateExpectation(element_ref, f'element.state.{state_name}', 'equals',
                    state_value) for state_name, state_value in expected_input_states.items())), parameters={
                    'element_id': element.element_id}))

        if 'clear_verified_text' in supported_by_element.get(element.element_id, set()):
            clear_expectations = [StateExpectation(element_ref, 'element.state.value', 'equals', '')]
            if element.states.get('ime_preedit_text'):
                clear_expectations.append(StateExpectation(element_ref, 'element.state.ime_preedit_text', 'absent'))
            if companion_text:
                candidates.append(_candidate(action_kind='clear_verified_text', expectations=tuple(
                    clear_expectations), parameters={'element_id': element.element_id,
                    'text_transport': 'companion_ime',
                    'input_field_id': str(element.states.get('input_field_id') or ''),
                    'prior_input_value': str(element.states.get('value') or ''), 'expected_input_value': ''}))
            else:
                append_element_candidate(element, 'clear_verified_text', clear_expectations)

        append_element_candidate(element, 'dismiss_overlay', (StateExpectation(surface_ref,
            'surface.overlay_present', 'equals', False),))
        for action_kind in ('long_press', 'double_tap'):
            append_element_candidate(element, action_kind, (StateExpectation(element_ref,
                'element.state.interaction_result', 'changed'),), exploratory=True)

    source_roles = {"drag_source", "source", "item"}
    destination_roles = {"drag_destination", "destination", "target"}
    unique_entity_elements = {entity_id: matches[0] for entity_id,
        matches in exact_elements_by_entity.items() if len(matches) == 1}
    source_elements = [unique_entity_elements[entity.entity_id] for entity
        in semantic_ir.entities if entity.role in source_roles and entity.entity_id in unique_entity_elements]
    destination_elements = [unique_entity_elements[entity.entity_id] for entity
        in semantic_ir.entities if entity.role in destination_roles and entity.entity_id in unique_entity_elements]
    if len(source_elements) == 1 and len(destination_elements) == 1:
        source_element = source_elements[0]
        destination_element = destination_elements[0]
        if source_element.element_id != destination_element.element_id:
            source_ref = _element_ref(source_element.element_id)
            destination_ref = _element_ref(destination_element.element_id)
            if ('drag' in supported_by_element.get(source_element.element_id, set())
                and 'drag' in supported_by_element.get(destination_element.element_id, set())):
                candidates.append(_candidate(action_kind='drag', expectations=(StateExpectation(source_ref,
                    'element.state.location_relation', 'equals', destination_ref),), parameters={
                    'source_element_id': source_element.element_id,
                    'destination_element_id': destination_element.element_id}))

    target_swipe_ref = _element_ref(target_swipe_presence.element_id) if target_swipe_presence is not None else ''
    system_specs = [('open_recent_apps', (StateExpectation(surface_ref, 'surface.kind', 'equals',
        expected_idempotent_system_surface_kind('open_recent_apps')),), False, {}), ('home',
        (StateExpectation(surface_ref, 'surface.kind', 'equals', expected_idempotent_system_surface_kind('home')),),
        False, {}), ('reveal_system_navigation', (StateExpectation(surface_ref, 'system_ui.navigation_bar_visible',
        'equals', True),), False, {}), ('back', (StateExpectation(surface_ref, 'surface.navigation_depth', 'changed'),),
        True, {}), ('wait_for_change', (StateExpectation(surface_ref, 'observation.changed', 'changed'),), True, {})]
    for direction in swipe_directions:
        anchored = target_swipe_presence is not None
        swipe_subject = target_swipe_ref if anchored else surface_ref
        swipe_predicate = "element.exists" if anchored else "surface.viewport"
        swipe_operator = "absent" if anchored else "changed"
        system_specs.append(('swipe', (StateExpectation(swipe_subject, swipe_predicate, swipe_operator),), not anchored,
            {'direction': direction, **({'element_id': target_swipe_presence.element_id} if anchored else {})}))
    for (action_kind, expectations, exploratory, parameters) in system_specs:
        if action_kind not in supported_system_actions:
            continue
        idempotent_surface_kind = expected_idempotent_system_surface_kind(action_kind)
        if idempotent_surface_kind is not None and _surface_kind(scene) == idempotent_surface_kind:
            continue
        if action_kind == 'reveal_system_navigation' and scene.system_ui.navigation_bar_visible is True:
            continue
        candidates.append(_candidate(action_kind=action_kind, expectations=expectations, exploratory=exploratory,
            parameters=parameters))

    action_priority = {'launch_app': 0, 'tap_semantic': 1, 'input_verified_text': 2, 'press_enter': 3,
        'clear_verified_text': 4, 'dismiss_overlay': 5, 'double_tap': 6, 'long_press': 7, 'drag': 8,
        'open_recent_apps': 9, 'home': 10, 'reveal_system_navigation': 11, 'back': 12, 'swipe': 13,
        'wait_for_change': 14}
    element_by_id = {item.element_id: item for item in sorted_elements}
    surfaces_by_id = {item.surface_id: item for item in semantic_ir.surfaces}
    target_surface = surfaces_by_id.get(active_subgoal.surface_ref)
    current_surface_kind = _surface_kind(scene)
    target_is_app = target_surface is not None and target_surface.kind == "app"
    wrong_app_surface = bool(target_is_app and current_surface_kind != 'launcher'
        and (not scene_matches_target_app_surface(scene, target_surface)))
    if ('launch_app' in available and target_is_app and launch_target is not None
        and not scene_matches_target_app_surface(scene, target_surface)):
        assert target_surface is not None
        candidates = [_candidate(action_kind='launch_app', expectations=(StateExpectation(surface_ref,
            'surface.active_ref', 'equals', target_surface.surface_id),), parameters={
            'target_surface_id': target_surface.surface_id, 'target_app_id': target_surface.app_id,
            'target_app_name': target_surface.app_name, 'launch_ref': launch_target['launch_ref'],
            'expected_app_id': launch_target['expected_app_id']})]
    unique_candidates = {item.candidate_id: item for item in candidates}
    required_system_action = {'launcher': 'home', 'recent_tasks': 'open_recent_apps'}.get(
        target_surface.kind if target_surface is not None else '')
    input_auxiliary_meanings = {'ime_exact_candidate', 'input_exact_literal_key', 'input_exact_enter_key',
        'switch_keyboard_layout', 'switch_keyboard_case', 'switch_keyboard_input_mode', 'input_next_field_key'}
    navigation_actions = {'back', 'open_recent_apps', 'swipe', 'reveal_system_navigation', 'wait_for_change'}

    def candidate_element(candidate: CanonicalActionCandidate) -> UIElement | None:
        return element_by_id.get(str(candidate.parameters.get("element_id") or ""))

    def clear_input_disposition(candidate: CanonicalActionCandidate) -> tuple[bool, bool]:
        """Return (must_clear_nonprefix, preserves_useful_preedit)."""

        element = candidate_element(candidate)
        if candidate.action_kind != 'clear_verified_text' or element is None or len(active_input_payload_entities) != 1:
            return False, False
        current_value = element.states.get("value")
        current_preedit = element.states.get("ime_preedit_text")
        exact_candidate = element.states.get("ime_exact_candidate_text")
        authorized_value = active_input_payload_entities[0].value
        if not isinstance(current_value, str) or not isinstance(authorized_value, str):
            return False, False
        useful_preedit = bool(isinstance(current_preedit, str) and current_preedit and isinstance(exact_candidate,
            str) and exact_candidate and authorized_value.startswith(current_value + exact_candidate))
        must_clear = not useful_preedit if isinstance(current_preedit,
            str) and current_preedit else bool(current_value and (not authorized_value.startswith(current_value)))
        return must_clear, useful_preedit

    def has_relation(candidate: CanonicalActionCandidate, relation_kind: str, object_refs: Iterable[str]) -> bool:
        element = candidate_element(candidate)
        if element is None:
            return False
        allowed_refs = frozenset(object_refs)
        return any((kind == relation_kind and value in allowed_refs for kind,
            value in bindings_by_element[element.element_id]))

    def matches_required_action(candidate: CanonicalActionCandidate) -> bool:
        action_kind = candidate.action_kind
        if action_kind == 'launch_app':
            return target_is_app and active_subgoal.external_impact == 'navigation_only' and not (
                active_required_actions - {'tap_semantic', 'launch_app'})
        must_clear, useful_preedit = clear_input_disposition(candidate)
        if (useful_preedit and (not ('clear_verified_text' in active_required_actions and 'input_verified_text' not
            in active_required_actions))):
            return False
        if not active_required_actions:
            return True
        if 'input_verified_text' not in active_required_actions:
            return action_kind in active_required_actions
        if action_kind == 'input_verified_text':
            return True
        if action_kind == 'press_enter':
            return "press_enter" in active_required_actions
        if action_kind == 'clear_verified_text':
            return "clear_verified_text" in active_required_actions or must_clear
        if action_kind != 'tap_semantic':
            return False
        element = candidate_element(candidate)
        return bool(element is not None and (element.role == 'input' or element.meaning in input_auxiliary_meanings))

    def tap_belongs(candidate: CanonicalActionCandidate) -> bool:
        element = candidate_element(candidate)
        if element is None:
            return False
        if wrong_app_surface:
            return False
        if element.meaning == 'input_next_field_key':
            return has_relation(candidate, 'binds_next_input_field', (item.field_id for item in active_input_fields))
        if element.meaning in input_auxiliary_meanings:
            return bool(element.meaning != 'input_exact_enter_key' and active_input_payload_refs)
        if element.role == 'input':
            return bool(element.states.get('focused') is not True and matches_active_input_field(element,
                ambiguous=True))
        if target_is_app:
            if current_surface_kind == 'launcher':
                return has_relation(candidate, 'binds_surface', (active_subgoal.surface_ref,))
            if wrong_app_surface:
                return False
        return bool(has_relation(candidate, 'binds_surface', (active_subgoal.surface_ref,)) or has_relation(candidate,
            'exact_literal_match', active_entity_refs) or (active_subgoal.external_impact == 'navigation_only'
            and element.meaning not in _ALL_EFFECT_CONTROL_MEANINGS))

    def belongs_to_active_subgoal(candidate: CanonicalActionCandidate) -> bool:
        if candidate.effect_ref:
            return candidate.effect_ref in active_effect_refs
        action_kind = candidate.action_kind
        if required_system_action is not None and current_surface_kind != target_surface.kind:
            return bool(active_subgoal.external_impact == 'navigation_only' and action_kind == required_system_action)
        if not matches_required_action(candidate):
            return False
        if action_kind in {'input_verified_text', 'press_enter', 'clear_verified_text'}:
            if action_kind == 'clear_verified_text':
                return bool(active_input_payload_refs) or bool('clear_verified_text' in active_required_actions
                    and active_targets_input)
            if action_kind == 'press_enter':
                element = candidate_element(candidate)
                field_id = str(element.states.get('input_field_id') if element is not None else '').strip()
                return bool(active_input_payload_refs or ('press_enter' in active_required_actions
                    and len(enter_input_field_ids) == 1 and (field_id in enter_input_field_ids)))
            return bool(active_input_payload_refs)
        if action_kind == 'tap_semantic':
            return tap_belongs(candidate)
        if action_kind == 'launch_app':
            return bool(target_is_app and candidate.parameters.get('target_surface_id') == active_subgoal.surface_ref)
        if action_kind == 'dismiss_overlay':
            return active_subgoal.external_impact == "navigation_only"
        if action_kind in {'double_tap', 'long_press', 'drag'}:
            return action_kind in active_required_actions
        if action_kind == 'home':
            if target_surface is None:
                return False
            if target_surface.kind == 'launcher':
                return current_surface_kind != "launcher"
            return wrong_app_surface
        if action_kind in navigation_actions:
            if active_input_fields:
                return action_kind in active_required_actions
            if wrong_app_surface:
                return False
            allowed_impact = {'read_only', 'navigation_only'} if action_kind == 'wait_for_change' else {
                'navigation_only'}
            return active_subgoal.external_impact in allowed_impact
        return False

    unique_candidates = {candidate_id: candidate for candidate_id,
        candidate in unique_candidates.items() if belongs_to_active_subgoal(candidate)}
    candidates = sorted(unique_candidates.values(),
        key=lambda item: (action_priority[item.action_kind], item.candidate_id))[:MAX_READY_CANDIDATES]
    status = "ready" if len(candidates) >= MIN_READY_CANDIDATES else "blocked"
    warnings: list[str] = []
    if status == 'blocked':
        warnings.append("insufficient_local_candidates")
    if any((len(matches) > 1 for matches in exact_elements_by_entity.values())):
        warnings.append("duplicate_exact_literal_binding")

    report = CanonicalActionCatalog(task_id=semantic_ir.task_id, device_id=semantic_ir.device_id,
        revision=semantic_ir.revision, candidates=tuple(candidates), status=status,
        warnings=tuple(sorted(set(warnings))))
    report.validate()
    return report


def canonical_candidate_expected_result(candidate: CanonicalActionCandidate, scene: UIScene) -> dict[str, Any]:
    """Project one canonical transition into the controller's visual result shape."""

    candidate.validate()
    if candidate.action_kind == 'launch_app':
        target_app_id = str(candidate.parameters.get('target_app_id') or '').strip()
        reject_if(not target_app_id, CanonicalActionProtocolError("App 直启候选缺少 typed 目标 App。"))
        return {'app_id': target_app_id}
    if candidate.action_kind == 'reveal_system_navigation':
        return {"system_ui": {"navigation_bar_visible": True}}
    if candidate.action_kind == 'swipe':
        element_id = str(candidate.parameters.get("element_id") or "").strip()
        if element_id:
            matches = tuple((element for element in scene.elements if element.element_id == element_id))
            reject_if(len(matches) != 1, CanonicalActionProtocolError('元素绑定 swipe 引用了不存在或不唯一的当前元素。'))
            element = matches[0]
            return {'content_changed': True, 'element_absent': {'element_id': element.element_id,
                'meaning': element.meaning, 'role': element.role, 'label': element.label}}
        return {"content_changed": True}

    element_by_ref = {_element_ref(element.element_id): element for element in scene.elements}
    state_expectations = [item for item in candidate.transition.expectations if item.predicate.startswith(
        'element.state.') and item.operator == 'equals' and (item.subject_ref in element_by_ref)]
    if state_expectations:
        subjects = {item.subject_ref for item in state_expectations}
        reject_if(len(subjects) != 1, CanonicalActionProtocolError('canonical candidate 包含多个元素的状态结果，无法形成唯一验证目标。'))
        subject_ref = next(iter(subjects))
        element = element_by_ref[subject_ref]
        states = {item.predicate.removeprefix('element.state.'): item.value for item in state_expectations}
        return {'element_state': {'meaning': element.meaning, 'states': states}}
    return {"scene_changed": True}
