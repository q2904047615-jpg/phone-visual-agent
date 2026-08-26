"""Pure domain service resolving and verifying canonical UI actions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
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
    scene_surface_kind,
)


UNIVERSAL_CONTROLLER_PROTOCOL_VERSION = "2026-08-26-universal-action-v17"

LOCAL_POINT_GROUNDING_SOURCE = "2026-08-25-stable-local-ocr-label-v1"
MAX_LOCAL_GROUNDING_BOX_GAP = 0.06

REVEAL_SYSTEM_NAVIGATION_EFFECT = {
    "system_ui": {"navigation_bar_visible": True}
}

# The seller batch accepts only the deterministic lowercase-and-space fragment
# minted by verified_text_transaction. Digits, uppercase and symbols continue
# through their independently audited visible-key paths.
GESTURE_EDGE_MARGIN = 0.02
TARGETED_SWIPE_EDGE_MARGIN = 0.08
MIN_DRAG_DISTANCE = 0.08
MAX_DRAG_DISTANCE = 0.90
DRAG_DURATION_SECONDS = 0.8
MIN_DRAG_RESULT_DISPLACEMENT = 0.04


class UniversalActionError(RuntimeError):
    pass


@dataclass(frozen=True)
class LocalPointGrounding:
    """Stable local evidence refining one already-selected target point."""

    source: str
    scene_fingerprint: str
    element_id: str
    label: str
    model_bounds: tuple[float, float, float, float]
    proposed_point: tuple[float, float]
    grounded_bounds: tuple[float, float, float, float]
    grounded_point: tuple[float, float]
    matched_frames: int
    inspected_frames: int

    @staticmethod
    def _validate_point(
        value: tuple[float, float],
        *,
        label: str,
    ) -> tuple[float, float]:
        if (
            not isinstance(value, tuple)
            or len(value) != 2
            or any(
                isinstance(part, bool)
                or not isinstance(part, (int, float))
                or not math.isfinite(float(part))
                for part in value
            )
        ):
            raise UniversalActionError(f"{label}格式无效。")
        point = (float(value[0]), float(value[1]))
        if not all(0.0 <= part <= 1.0 for part in point):
            raise UniversalActionError(f"{label}超出归一化画面。")
        return point

    @staticmethod
    def _validate_bounds(
        value: tuple[float, float, float, float],
        *,
        label: str,
    ) -> tuple[float, float, float, float]:
        if (
            not isinstance(value, tuple)
            or len(value) != 4
            or any(
                isinstance(part, bool)
                or not isinstance(part, (int, float))
                or not math.isfinite(float(part))
                for part in value
            )
        ):
            raise UniversalActionError(f"{label}格式无效。")
        bounds = tuple(float(part) for part in value)
        left, top, right, bottom = bounds
        if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
            raise UniversalActionError(f"{label}超出归一化画面。")
        return bounds

    def validate_for(self, scene: UIScene, element: UIElement) -> None:
        if self.source != LOCAL_POINT_GROUNDING_SOURCE:
            raise UniversalActionError("本地落点证据来源无效。")
        if self.scene_fingerprint != scene.fingerprint:
            raise UniversalActionError("本地落点证据不属于当前新鲜画面。")
        if self.element_id != element.element_id or self.label != element.label:
            raise UniversalActionError("本地落点证据没有绑定当前唯一目标。")
        if element.role not in {
            "button",
            "icon",
            "text",
            "tab",
            "toggle",
            "image",
            "list_item",
        }:
            raise UniversalActionError("当前目标类型不允许使用文字落点修正。")
        if (
            element.element_id.startswith("local_audited_")
            or element.meaning == "application_text_input"
            or element.meaning.startswith(("input_", "ime_", "switch_keyboard_"))
        ):
            raise UniversalActionError("输入事务目标不允许使用普通文字落点修正。")
        model_bounds = self._validate_bounds(self.model_bounds, label="模型目标框")
        grounded_bounds = self._validate_bounds(
            self.grounded_bounds,
            label="本地文字框",
        )
        proposed = self._validate_point(
            self.proposed_point,
            label="模型提议落点",
        )
        grounded = self._validate_point(
            self.grounded_point,
            label="本地修正落点",
        )
        if any(
            abs(actual - expected) > 1e-9
            for actual, expected in zip(model_bounds, element.bounds)
        ) or any(
            abs(actual - expected) > 1e-9
            for actual, expected in zip(proposed, element.center)
        ):
            raise UniversalActionError("本地落点证据与当前目标几何不一致。")
        left, top, right, bottom = grounded_bounds
        if not (left <= grounded[0] <= right and top <= grounded[1] <= bottom):
            raise UniversalActionError("本地修正落点不在已识别文字框内。")
        if (
            isinstance(self.matched_frames, bool)
            or isinstance(self.inspected_frames, bool)
            or not isinstance(self.matched_frames, int)
            or not isinstance(self.inspected_frames, int)
            or not (2 <= self.matched_frames <= self.inspected_frames <= 4)
        ):
            raise UniversalActionError("本地落点证据缺少至少两帧稳定匹配。")
        model_left, model_top, model_right, model_bottom = model_bounds
        grounded_left, grounded_top, grounded_right, grounded_bottom = (
            grounded_bounds
        )
        horizontal_gap = max(
            0.0,
            grounded_left - model_right,
            model_left - grounded_right,
        )
        vertical_gap = max(
            0.0,
            grounded_top - model_bottom,
            model_top - grounded_bottom,
        )
        if math.hypot(horizontal_gap, vertical_gap) > MAX_LOCAL_GROUNDING_BOX_GAP:
            raise UniversalActionError("本地文字框与模型目标框不属于同一邻近区域。")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "scene_fingerprint": self.scene_fingerprint,
            "element_id": self.element_id,
            "label": self.label,
            "model_bounds": list(self.model_bounds),
            "proposed_point": list(self.proposed_point),
            "grounded_bounds": list(self.grounded_bounds),
            "grounded_point": list(self.grounded_point),
            "matched_frames": self.matched_frames,
            "inspected_frames": self.inspected_frames,
        }


@dataclass(frozen=True)
class ResolvedSemanticAction:
    """One device-independent action resolved from one fresh scene."""

    node_id: str
    kind: str
    normalized_point: tuple[float, float] | None = None
    normalized_end_point: tuple[float, float] | None = None
    text: str | None = None
    input_fragment: str | None = None
    input_method: str | None = None
    input_pinyin: str | None = None
    prior_input_value: str | None = None
    expected_input_value: str | None = None
    delete_count: int | None = None
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
    proposed_normalized_point: tuple[float, float] | None = None
    point_grounding: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.normalized_point is not None:
            value["normalized_point"] = list(self.normalized_point)
        if self.normalized_end_point is not None:
            value["normalized_end_point"] = list(self.normalized_end_point)
        if self.proposed_normalized_point is not None:
            value["proposed_normalized_point"] = list(
                self.proposed_normalized_point
            )
        else:
            value.pop("proposed_normalized_point", None)
        if self.point_grounding is None:
            value.pop("point_grounding", None)
        return value


class UniversalActionController:
    """Resolve semantic actions without knowing WeChat, Douyin or any App UI."""

    def __init__(
        self,
        *,
        min_confidence: float = MIN_TARGET_CONFIDENCE,
    ) -> None:
        self.min_confidence = float(min_confidence)

    def resolve_one(
        self,
        action: SemanticAction,
        scene: UIScene,
        *,
        confirmed: bool = False,
        local_point_grounding: LocalPointGrounding | None = None,
    ) -> ResolvedSemanticAction:
        scene.validate()
        if local_point_grounding is not None and action.action not in {
            "tap_semantic",
            "dismiss_overlay",
            "double_tap",
            "long_press",
        }:
            raise UniversalActionError(
                "本地文字落点只能修正当前已选中的普通点按目标。"
            )
        if not scene.stable:
            raise UniversalActionError("页面仍在变化，不能执行动作。")
        if float(scene.confidence) < self.min_confidence:
            target_local_candidate = scene.unique_trusted_goal_element(
                min_confidence=self.min_confidence
            )
            action_element_id = str(action.params.get("element_id") or "").strip()
            if (
                action.action
                not in {
                    "tap_semantic",
                    "dismiss_overlay",
                    "input_verified_text",
                    "press_enter",
                    "clear_verified_text",
                    "double_tap",
                    "long_press",
                }
                or target_local_candidate is None
                or target_local_candidate.element_id != action_element_id
            ):
                raise UniversalActionError(
                    "页面整体置信度不足，且没有唯一可信的目标局部证据。"
                )
        formal_candidate_id = str(
            action.params.get("formal_candidate_id") or ""
        ).strip()
        formal_transition = dict(action.params.get("formal_transition") or {})
        expected_effect = dict(action.params.get("expected_effect") or {})
        if "system_ui" in expected_effect and action.action != "reveal_system_navigation":
            raise UniversalActionError(
                "结构化 system_ui 后置条件只允许用于系统导航栏唤出动作。"
            )

        if action.action == "tap_semantic":
            element = self._resolve_target(action, scene)
            if element.states.get("local_text_clear") is True:
                self._validate_local_text_clear(
                    element,
                    scene,
                    formal=bool(formal_candidate_id),
                )
            if element.meaning in {
                "input_exact_literal_key",
                "switch_keyboard_layout",
                "switch_keyboard_case",
                "switch_keyboard_input_mode",
            }:
                self._validate_input_auxiliary_tap(
                    element,
                    scene,
                    expected_effect,
                    formal=bool(formal_candidate_id),
                )
            if element.meaning == "input_next_field_key":
                self._validate_next_field_tap(
                    element,
                    formal_transition,
                    expected_effect,
                    formal=bool(formal_candidate_id),
                )
            resolved = self._point_action(
                action,
                element,
                expected_effect,
                scene.fingerprint,
                scene=scene,
                local_point_grounding=local_point_grounding,
            )
            if element.meaning in {
                "ime_exact_candidate",
                "input_exact_literal_key",
                "switch_keyboard_layout",
                "switch_keyboard_case",
                "switch_keyboard_input_mode",
            }:
                expected_element = expected_effect.get("element_state")
                expected_states = (
                    expected_element.get("states")
                    if isinstance(expected_element, dict)
                    else None
                )
                expected_value = (
                    expected_states.get("value")
                    if isinstance(expected_states, dict)
                    else None
                )
                if not isinstance(expected_value, str):
                    raise UniversalActionError(
                        "输入辅助键缺少精确输入值后置条件。"
                    )
                resolved = replace(
                    resolved,
                    prior_input_value=element.states.get("prior_input_value"),
                    expected_input_value=expected_value,
                    input_element_id=element.states.get("input_element_id"),
                )
            return resolved
        if action.action == "press_enter":
            element = self._resolve_target(action, scene)
            if element.meaning != "input_exact_enter_key":
                raise UniversalActionError(
                    "press_enter 必须绑定本地审计的唯一换行键。"
                )
            self._validate_input_auxiliary_tap(
                element,
                scene,
                expected_effect,
                formal=bool(formal_candidate_id),
            )
            resolved = self._point_action(
                action,
                element,
                expected_effect,
                scene.fingerprint,
            )
            return replace(
                resolved,
                prior_input_value=element.states.get("prior_input_value"),
                expected_input_value=element.states.get("expected_input_value"),
                input_element_id=element.states.get("input_element_id"),
            )
        if action.action == "dismiss_overlay":
            if not action.params.get("target"):
                action = SemanticAction(
                    node_id=action.node_id,
                    action=action.action,
                    params={**action.params, "target": "close"},
                )
            element = self._resolve_target(action, scene)
            return self._point_action(
                action,
                element,
                expected_effect,
                scene.fingerprint,
                scene=scene,
                local_point_grounding=local_point_grounding,
            )
        if action.action == "input_verified_text":
            text = str(action.params.get("text") or "")
            element = self._resolve_target(action, scene, required_role="input")
            if element.states.get("focused") is not True:
                raise UniversalActionError("文字输入前必须有当前画面证明输入框已聚焦。")
            if element.states.get("keyboard_layout") != "qwerty":
                raise UniversalActionError("精确文字输入要求当前画面确认 QWERTY 键盘。")
            if (
                not formal_candidate_id
                and element.states.get("goal_relevant") is not True
            ):
                raise UniversalActionError("文字输入目标必须由当前画面证明与当前目标相关。")
            try:
                input_step = plan_from_input_states(text, element.states)
            except (ValueError, VerifiedTextTransactionError) as exc:
                raise UniversalActionError(f"无法建立精确文字输入事务：{exc}") from exc
            if input_step is None:
                raise UniversalActionError("输入框已经逐字等于目标文字，不得重复输入。")
            if input_step.kind == "literal_key":
                raise UniversalActionError(
                    "下一分段需要独立可见的数字、空格或符号键审计，不能按字母键盘猜测。"
                )
            if element.states.get("keyboard_input_mode") != input_step.required_mode:
                raise UniversalActionError(
                    "当前键盘输入模式与下一确定性文字分段不一致。"
                )
            if (
                input_step.required_case_mode
                and element.states.get("keyboard_case_mode")
                != input_step.required_case_mode
            ):
                raise UniversalActionError(
                    "当前键盘大小写状态与下一确定性英文分段不一致。"
                )
            if element.states.get("ime_preedit_text"):
                raise UniversalActionError("当前仍有未完成的输入法组合，禁止继续键入。")
            expected_states = (
                {
                    "value": input_step.current_text,
                    "ime_preedit_text": input_step.pinyin,
                    "ime_exact_candidate_text": input_step.segment,
                }
                if input_step.kind == "chinese_pinyin"
                else {"value": input_step.expected_value}
            )
            if (
                "element_state" not in expected_effect
                and input_step.kind == "direct_latin"
                and input_step.current_text == ""
                and is_direct_latin_segment(input_step.segment)
            ):
                # Backward-compatible stage-1 authority: the first certified
                # profile already bound an empty direct-Latin field and exact
                # lowercase segment. Derive, rather than guess, its postcondition.
                expected_effect = {
                    **expected_effect,
                    "element_state": {
                        "meaning": element.meaning,
                        "states": expected_states,
                    },
                }
            if expected_effect.get("element_state") != {
                "meaning": element.meaning,
                "states": expected_states,
            }:
                raise UniversalActionError(
                    "精确文字输入的后置条件没有绑定下一确定性分段。"
                )
            eligible_inputs = tuple(
                candidate
                for candidate in scene.elements
                if candidate.role == "input"
                and float(candidate.confidence) >= self.min_confidence
                and candidate.states.get("visible") is not False
                and (
                    candidate.element_id == element.element_id
                    if formal_candidate_id
                    else candidate.states.get("goal_relevant") is True
                )
                and candidate.states.get("focused") is True
                and candidate.states.get("keyboard_layout") == "qwerty"
                and candidate.states.get("keyboard_input_mode")
                == input_step.required_mode
                and (
                    not input_step.required_case_mode
                    or candidate.states.get("keyboard_case_mode")
                    == input_step.required_case_mode
                )
                and not candidate.states.get("ime_preedit_text")
            )
            if len(eligible_inputs) != 1 or eligible_inputs[0].element_id != element.element_id:
                raise UniversalActionError(
                    "精确文字输入要求当前画面只有一个符合安全条件的目标输入框。"
                )
            return ResolvedSemanticAction(
                node_id=action.node_id,
                kind="input_verified_text",
                normalized_point=element.center,
                text=text,
                input_fragment=input_step.segment,
                input_method=input_step.kind,
                input_pinyin=input_step.pinyin or None,
                prior_input_value=input_step.current_text,
                expected_input_value=input_step.expected_value,
                target_element_id=element.element_id,
                before_fingerprint=scene.fingerprint,
                expected_effect=expected_effect,
                formal_candidate_id=formal_candidate_id,
                formal_transition=formal_transition,
            )
        if action.action == "clear_verified_text":
            element = self._resolve_target(action, scene, required_role="input")
            observed_value = element.states.get("value")
            observed_preedit = element.states.get("ime_preedit_text", "")
            if element.states.get("focused") is not True:
                raise UniversalActionError("清空文字前必须有当前画面证明输入框已聚焦。")
            if not isinstance(observed_value, str):
                raise UniversalActionError("清空文字要求当前画面提供精确 states.value。")
            if not isinstance(observed_preedit, str):
                raise UniversalActionError("清空文字的输入法预编辑状态格式无效。")
            extra_delete_units = element.states.get("clear_extra_delete_units", 0)
            if (
                isinstance(extra_delete_units, bool)
                or not isinstance(extra_delete_units, int)
                or not 0 <= extra_delete_units <= 30
            ):
                raise UniversalActionError("清空文字的额外视觉行退格单位无效。")
            if not observed_value and not observed_preedit:
                raise UniversalActionError("清空文字要求应用值或输入法预编辑至少一项非空。")
            delete_count = (
                editable_character_count(observed_value)
                + editable_character_count(observed_preedit)
                + extra_delete_units
            )
            if not 1 <= delete_count <= 100:
                raise UniversalActionError("清空文字的已验证字符数必须在1～100之间。")
            if (
                not formal_candidate_id
                and element.states.get("goal_relevant") is not True
            ):
                raise UniversalActionError("清空文字目标必须由当前画面证明与当前目标相关。")
            eligible_inputs = tuple(
                candidate
                for candidate in scene.elements
                if candidate.role == "input"
                and float(candidate.confidence) >= self.min_confidence
                and candidate.states.get("visible") is not False
                and (
                    candidate.element_id == element.element_id
                    if formal_candidate_id
                    else candidate.states.get("goal_relevant") is True
                )
                and candidate.states.get("focused") is True
                and isinstance(candidate.states.get("value"), str)
                and isinstance(candidate.states.get("ime_preedit_text", ""), str)
                and (
                    bool(candidate.states.get("value"))
                    or bool(candidate.states.get("ime_preedit_text"))
                )
            )
            if len(eligible_inputs) != 1 or eligible_inputs[0].element_id != element.element_id:
                raise UniversalActionError(
                    "清空文字要求当前画面只有一个符合安全条件的非空目标输入框。"
                )
            expected_state = expected_effect.get("element_state")
            if not isinstance(expected_state, dict):
                raise UniversalActionError("清空文字必须声明输入框空值后置条件。")
            if (
                str(expected_state.get("meaning") or "") != element.meaning
                or expected_state.get("states") != {"value": ""}
            ):
                raise UniversalActionError("清空文字的后置条件必须精确绑定原输入框空值。")
            return ResolvedSemanticAction(
                node_id=action.node_id,
                kind="clear_verified_text",
                normalized_point=element.center,
                text="",
                delete_count=delete_count,
                target_element_id=element.element_id,
                before_fingerprint=scene.fingerprint,
                expected_effect=expected_effect,
                formal_candidate_id=formal_candidate_id,
                formal_transition=formal_transition,
            )
        if action.action == "double_tap":
            element = self._resolve_target(action, scene)
            self._validate_gesture_point(element.center, label="双击落点")
            self._require_visual_postcondition(
                "double_tap",
                expected_effect,
                scene,
            )
            return self._point_action(
                action,
                element,
                expected_effect,
                scene.fingerprint,
                scene=scene,
                local_point_grounding=local_point_grounding,
            )
        if action.action == "long_press":
            element = self._resolve_target(action, scene)
            self._validate_gesture_point(element.center, label="长按落点")
            duration_ms = action.params.get("duration_ms", 800)
            if isinstance(duration_ms, bool) or not isinstance(duration_ms, (int, float)):
                raise UniversalActionError("长按 duration_ms 格式无效。")
            if not 500 <= float(duration_ms) <= 2000:
                raise UniversalActionError("长按 duration_ms 必须在500～2000之间。")
            self._require_visual_postcondition(
                "long_press",
                expected_effect,
                scene,
            )
            resolved = self._point_action(
                action,
                element,
                expected_effect,
                scene.fingerprint,
                scene=scene,
                local_point_grounding=local_point_grounding,
            )
            return replace(
                resolved,
                hold_seconds=float(duration_ms) / 1000.0,
            )
        if action.action == "drag":
            source = self._resolve_target(action, scene, prefix="source_")
            destination = self._resolve_target(
                action,
                scene,
                prefix="destination_",
            )
            if source.element_id == destination.element_id:
                raise UniversalActionError("拖动起点和终点不能是同一元素。")
            source_container_error = compact_drag_source_container_error(scene, source)
            if source_container_error:
                raise UniversalActionError(source_container_error)
            self._validate_gesture_point(source.center, label="拖动起点")
            self._validate_gesture_point(destination.center, label="拖动终点")
            distance = math.dist(source.center, destination.center)
            if not MIN_DRAG_DISTANCE <= distance <= MAX_DRAG_DISTANCE:
                raise UniversalActionError(
                    "拖动两端中心距离必须在"
                    f"{MIN_DRAG_DISTANCE:.2f}～{MAX_DRAG_DISTANCE:.2f}个归一化屏幕单位之间。"
                )
            self._require_visual_postcondition("drag", expected_effect, scene)
            return ResolvedSemanticAction(
                node_id=action.node_id,
                kind="drag",
                normalized_point=source.center,
                normalized_end_point=destination.center,
                target_element_id=source.element_id,
                destination_element_id=destination.element_id,
                hold_seconds=DRAG_DURATION_SECONDS,
                path_distance=distance,
                before_fingerprint=scene.fingerprint,
                expected_effect=expected_effect,
                formal_candidate_id=formal_candidate_id,
                formal_transition=formal_transition,
            )
        if action.action == "reveal_system_navigation":
            unexpected = set(action.params) - {
                "expected_effect",
                "formal_candidate_id",
                "formal_report_digest",
                "formal_transition",
            }
            if unexpected:
                raise UniversalActionError(
                    "系统导航栏唤出动作不能携带坐标、方向、距离或其他参数。"
                )
            self._require_hidden_immersive_navigation(scene)
            if expected_effect != REVEAL_SYSTEM_NAVIGATION_EFFECT:
                raise UniversalActionError(
                    "系统导航栏唤出动作必须精确声明导航栏可见后置条件。"
                )
            return ResolvedSemanticAction(
                node_id=action.node_id,
                kind="reveal_system_navigation",
                before_fingerprint=scene.fingerprint,
                expected_effect=expected_effect,
                formal_candidate_id=formal_candidate_id,
                formal_transition=formal_transition,
            )
        if action.action == "swipe":
            direction = str(action.params.get("direction") or "").strip().lower()
            if direction not in {"up", "down", "left", "right"}:
                raise UniversalActionError(f"不支持的滑动方向：{direction}")
            element_id = str(action.params.get("element_id") or "").strip()
            absence = expected_effect.get("element_absent")
            if element_id:
                element = self._resolve_target(action, scene)
                self._validate_targeted_swipe_absence_contract(
                    element,
                    absence,
                )
                start, end = self._targeted_swipe_path(element, direction)
                distance = math.dist(start, end)
                return ResolvedSemanticAction(
                    node_id=action.node_id,
                    kind="swipe",
                    normalized_point=start,
                    normalized_end_point=end,
                    direction=direction,
                    hold_seconds=DRAG_DURATION_SECONDS,
                    path_distance=distance,
                    target_element_id=element.element_id,
                    before_fingerprint=scene.fingerprint,
                    expected_effect=expected_effect,
                    formal_candidate_id=formal_candidate_id,
                    formal_transition=formal_transition,
                )
            if absence is not None:
                raise UniversalActionError(
                    "元素消失后置条件必须绑定同一 swipe element_id。"
                )
            return ResolvedSemanticAction(
                node_id=action.node_id,
                kind="swipe",
                direction=direction,
                before_fingerprint=scene.fingerprint,
                expected_effect=expected_effect,
                formal_candidate_id=formal_candidate_id,
                formal_transition=formal_transition,
            )
        if action.action in {
            "back",
            "home",
            "open_recent_apps",
            "observe",
            "wait_for_change",
            "verify",
            "finish",
        }:
            return ResolvedSemanticAction(
                node_id=action.node_id,
                kind=action.action,
                before_fingerprint=scene.fingerprint,
                expected_effect=expected_effect,
                formal_candidate_id=formal_candidate_id,
                formal_transition=formal_transition,
            )
        if action.action == "ensure_app":
            app_id = str(action.params.get("app_id") or "").strip().lower()
            if not app_id:
                raise UniversalActionError("ensure_app 缺少 app_id。")
            return ResolvedSemanticAction(
                node_id=action.node_id,
                kind="ensure_app",
                before_fingerprint=scene.fingerprint,
                expected_effect={"app_id": app_id, **expected_effect},
                formal_candidate_id=formal_candidate_id,
                formal_transition=formal_transition,
            )
        raise UniversalActionError(f"通用动作控制器尚不支持：{action.action}")

    def _validate_local_text_clear(
        self,
        element: UIElement,
        scene: UIScene,
        *,
        formal: bool = False,
    ) -> None:
        if (
            element.meaning != "clear_local_text"
            or element.role not in {"button", "icon"}
            or element.label.strip().casefold() not in {"×", "✕", "✖", "x"}
        ):
            raise UniversalActionError("本地文字清空必须绑定真实可见的独立 × 图形。")
        inputs = tuple(
            candidate
            for candidate in scene.elements
            if candidate.role == "input"
            and float(candidate.confidence) >= self.min_confidence
            and (formal or candidate.states.get("goal_relevant") is True)
            and candidate.states.get("focused") is True
            and isinstance(candidate.states.get("value"), str)
            and bool(candidate.states.get("value"))
            and candidate.states.get("keyboard_layout")
            in {"qwerty", "numeric", "symbol", "unknown"}
        )
        if len(inputs) != 1:
            raise UniversalActionError(
                "本地文字清空要求唯一非空、已聚焦且带软键盘事实的目标输入框。"
            )
        input_element = inputs[0]
        il, it, ir, ib = input_element.bounds
        el, et, er, eb = element.bounds
        element_height = max(1e-9, eb - et)
        input_height = max(1e-9, ib - it)
        vertical_overlap = max(0.0, min(ib, eb) - max(it, et))
        if not (
            vertical_overlap / element_height >= 0.6
            and el >= il + 0.4 * (ir - il)
            and er <= min(1.0, ir + 0.2)
            and max(0.0, el - ir) <= max(0.04, input_height)
            and er - el <= 2.0 * input_height
            and element_height <= 1.5 * input_height
        ):
            raise UniversalActionError("本地文字清空控件没有与唯一目标输入框形成可信几何绑定。")

    def _validate_next_field_tap(
        self,
        element: UIElement,
        formal_transition: dict[str, Any],
        expected_effect: dict[str, Any],
        *,
        formal: bool,
    ) -> None:
        states = element.states
        source_id = str(states.get("source_input_field_id") or "").strip()
        target_id = str(states.get("target_input_field_id") or "").strip()
        target_label = str(states.get("target_input_field_label") or "").strip()
        expectations = formal_transition.get("expectations")
        if not (
            formal
            and element.role == "button"
            and float(element.confidence) >= 0.9
            and states.get("fully_visible") is True
            and states.get("input_next_field_key") is True
            and states.get("key_action") == "next"
            and source_id
            and target_id
            and target_label
            and source_id != target_id
            and expected_effect == {"scene_changed": True}
            and isinstance(expectations, list)
            and expectations == [{
                "subject_ref": target_id,
                "predicate": "input_field.focused",
                "operator": "equals",
                "value": True,
            }]
        ):
            raise UniversalActionError("Next键没有绑定唯一typed字段切换合同。")

    def _validate_input_auxiliary_tap(
        self,
        element: UIElement,
        scene: UIScene,
        expected_effect: dict[str, Any],
        *,
        formal: bool = False,
    ) -> None:
        states = element.states
        if (
            element.role != "button"
            or float(element.confidence) < 0.9
            or (not formal and states.get("goal_relevant") is not True)
            or states.get("fully_visible") is not True
        ):
            raise UniversalActionError("输入辅助键缺少本轮完整、高置信本地审计。")
        input_id = str(states.get("input_element_id") or "").strip()
        try:
            input_element = scene.get_element(
                input_id,
                min_confidence=self.min_confidence,
            )
        except UISceneError as exc:
            raise UniversalActionError(f"输入辅助键没有绑定唯一输入框：{exc}") from exc
        prior_value = states.get("prior_input_value")
        if (
            input_element.role != "input"
            or input_element.states.get("focused") is not True
            or not isinstance(prior_value, str)
            or input_element.states.get("value") != prior_value
        ):
            raise UniversalActionError("输入辅助键与当前精确输入前缀不一致。")
        expected_state = expected_effect.get("element_state")
        if not isinstance(expected_state, dict) or expected_state.get("meaning") != input_element.meaning:
            raise UniversalActionError("输入辅助键没有绑定原输入框的后置条件。")
        expected_states = expected_state.get("states")
        if not isinstance(expected_states, dict):
            raise UniversalActionError("输入辅助键后置状态格式无效。")
        if element.meaning == "input_exact_literal_key":
            key_value = states.get("key_value")
            expected_value = states.get("expected_input_value")
            if (
                states.get("input_literal_key") is not True
                or not isinstance(key_value, str)
                or len(key_value) != 1
                or expected_value != prior_value + key_value
                or expected_states != {"value": expected_value}
            ):
                raise UniversalActionError("逐键输入没有绑定唯一下一字符和精确结果。")
        elif element.meaning == "input_exact_enter_key":
            key_value = states.get("key_value")
            expected_value = states.get("expected_input_value")
            if (
                states.get("input_enter_key") is not True
                or states.get("key_action") != "newline"
                or key_value != "\n"
                or expected_value != prior_value + "\n"
                or expected_states != {"value": expected_value}
            ):
                raise UniversalActionError(
                    "换行键没有绑定 multiline newline 与精确输入结果。"
                )
            if input_element.states.get("input_multiline") is not True:
                raise UniversalActionError("当前输入框没有本地多行字段凭据。")
        elif element.meaning == "switch_keyboard_layout":
            current = states.get("current_layout")
            target = states.get("target_layout")
            if (
                states.get("keyboard_layout_switch") is not True
                or input_element.states.get("keyboard_layout") != current
                or current not in {"qwerty", "numeric", "symbol"}
                or target not in {"qwerty", "numeric", "symbol"}
                or current == target
                or expected_states != {"value": prior_value, "keyboard_layout": target}
            ):
                raise UniversalActionError("键盘布局切换方向或后置条件无效。")
        elif element.meaning == "switch_keyboard_case":
            current = states.get("current_mode")
            target = states.get("target_mode")
            if (
                states.get("keyboard_case_switch") is not True
                or input_element.states.get("keyboard_layout") != "qwerty"
                or input_element.states.get("keyboard_input_mode") != "direct_latin"
                or input_element.states.get("keyboard_case_mode") != current
                or current not in {"lower", "upper"}
                or target not in {"lower", "upper"}
                or current == target
                or expected_states != {"value": prior_value, "keyboard_case_mode": target}
            ):
                raise UniversalActionError("键盘大小写切换方向或后置条件无效。")
        elif element.meaning == "switch_keyboard_input_mode":
            current = states.get("current_mode")
            target = states.get("target_mode")
            if (
                states.get("keyboard_input_mode_switch") is not True
                or input_element.states.get("keyboard_layout") != "qwerty"
                or input_element.states.get("keyboard_input_mode") != current
                or current not in {"direct_latin", "chinese_pinyin"}
                or target not in {"direct_latin", "chinese_pinyin"}
                or current == target
                or expected_states
                != {"value": prior_value, "keyboard_input_mode": target}
            ):
                raise UniversalActionError("键盘输入模式切换方向或后置条件无效。")

    def verify_after_action(
        self,
        resolved: ResolvedSemanticAction,
        before: UIScene,
        after: UIScene,
    ) -> None:
        before.validate()
        after.validate()
        if resolved.kind not in {"observe", "verify", "finish", "wait_for_change"}:
            if not resolved.before_fingerprint:
                raise UniversalActionError("动作缺少执行前场景 fingerprint。")
            if resolved.before_fingerprint != before.fingerprint:
                raise UniversalActionError("动作绑定的 fingerprint 已过期。")
        if resolved.kind in {"double_tap", "long_press", "drag"}:
            self._require_visual_postcondition(
                resolved.kind,
                resolved.expected_effect,
                before,
            )
        if resolved.kind == "long_press":
            self._verify_long_press_contract(resolved, before)
        if not after.stable or float(after.confidence) < self.min_confidence:
            raise UniversalActionError("动作后的页面不稳定或置信度不足。")
        if (
            resolved.kind not in {"observe", "verify", "finish", "wait_for_change"}
            and resolved.kind != "reveal_system_navigation"
            and resolved.expected_effect.get("allow_unchanged") is not True
            and (
                (
                    before.fingerprint
                    and after.fingerprint
                    and before.fingerprint == after.fingerprint
                )
                or (
                    resolved.kind != "drag"
                    and self.scenes_semantically_equivalent(before, after)
                )
            )
        ):
            raise UniversalActionError("动作后页面没有可验证的语义变化。")
        expected = resolved.expected_effect
        expected_app = str(expected.get("app_id") or "").strip()
        if expected_app and after.foreground_app_id != expected_app:
            raise UniversalActionError(
                "动作后前台 App 不符合预期："
                f"{after.foreground_app_id} != {expected_app}"
            )
        expected_screen = str(expected.get("screen_id") or "").strip()
        if expected_screen and after.screen_id != expected_screen:
            raise UniversalActionError(
                f"动作后页面不符合预期：{after.screen_id} != {expected_screen}"
            )
        if resolved.kind in {
            "input_verified_text",
            "press_enter",
            "clear_verified_text",
        }:
            self._verify_exact_input_value(resolved, before, after)
        if expected.get("element_absent") is not None:
            self._verify_expected_element_absent(resolved, before, after)
        if resolved.formal_candidate_id:
            self._verify_formal_transition(resolved, before, after)
        element_state = expected.get("element_state")
        if element_state is not None:
            if not isinstance(element_state, dict):
                raise UniversalActionError("expected_effect.element_state 格式无效。")
            meaning = str(element_state.get("meaning") or "").strip()
            states = dict(element_state.get("states") or {})
            try:
                after.resolve_unique(
                    meaning=meaning,
                    states=states,
                    min_confidence=self.min_confidence,
                )
            except UISceneError as exc:
                before_target_is_input = False
                if resolved.target_element_id:
                    try:
                        before_target_is_input = (
                            before.get_element(resolved.target_element_id).role
                            == "input"
                        )
                    except UISceneError:
                        before_target_is_input = False
                input_aliases = tuple(
                    element
                    for element in after.elements
                    if resolved.kind in {
                        "tap_semantic", "input_verified_text", "clear_verified_text"
                    }
                    and before_target_is_input
                    and element.role == "input"
                    and float(element.confidence) >= self.min_confidence
                    and all(element.states.get(key) == value for key, value in states.items())
                )
                provisional_direct_preedits = tuple(
                    element
                    for element in after.elements
                    if resolved.kind == "input_verified_text"
                    and resolved.input_method == "direct_latin"
                    and before_target_is_input
                    and element.role == "input"
                    and float(element.confidence) >= self.min_confidence
                    and self._is_exact_direct_latin_preedit_transition(
                        resolved,
                        after,
                        element,
                    )
                )
                if len(input_aliases) != 1 and len(provisional_direct_preedits) != 1:
                    raise UniversalActionError(
                        f"动作结果缺少元素状态证据：{exc}"
                    ) from exc
        if resolved.kind == "long_press":
            self._verify_long_press_result(resolved, before, after)
        if resolved.kind == "drag":
            self._verify_drag_result(resolved, before, after)
        if resolved.kind == "reveal_system_navigation":
            self._verify_revealed_system_navigation(resolved, before, after)

    @staticmethod
    def _require_hidden_immersive_navigation(scene: UIScene) -> Any:
        system_ui = getattr(scene, "system_ui", None)
        if system_ui is None:
            raise UniversalActionError(
                "系统导航栏唤出动作缺少结构化 scene.system_ui。"
            )
        immersive = getattr(system_ui, "immersive_or_fullscreen", None)
        navigation_visible = getattr(system_ui, "navigation_bar_visible", None)
        if isinstance(system_ui, Mapping):
            if immersive is None:
                immersive = system_ui.get("immersive_or_fullscreen")
            if navigation_visible is None:
                navigation_visible = system_ui.get("navigation_bar_visible")
        if (
            immersive is not True
            or navigation_visible is not False
        ):
            raise UniversalActionError(
                "系统导航栏唤出动作要求当前画面明确处于沉浸态且导航栏隐藏。"
            )
        return system_ui

    def _verify_revealed_system_navigation(
        self,
        resolved: ResolvedSemanticAction,
        before: UIScene,
        after: UIScene,
    ) -> None:
        self._require_hidden_immersive_navigation(before)
        if resolved.expected_effect != REVEAL_SYSTEM_NAVIGATION_EFFECT:
            raise UniversalActionError("系统导航栏唤出动作的结构化后置条件无效。")
        system_ui = getattr(after, "system_ui", None)
        navigation_visible = (
            getattr(system_ui, "navigation_bar_visible", None)
            if system_ui is not None
            else None
        )
        if isinstance(system_ui, Mapping) and navigation_visible is None:
            navigation_visible = system_ui.get("navigation_bar_visible")
        if navigation_visible is not True:
            raise UniversalActionError(
                "动作后缺少结构化导航栏可见证据。"
            )

    @staticmethod
    def _validate_gesture_point(point: tuple[float, float], *, label: str) -> None:
        x, y = point
        if not (
            GESTURE_EDGE_MARGIN <= x <= 1.0 - GESTURE_EDGE_MARGIN
            and GESTURE_EDGE_MARGIN <= y <= 1.0 - GESTURE_EDGE_MARGIN
        ):
            raise UniversalActionError(
                f"{label}必须离画面边缘至少{GESTURE_EDGE_MARGIN:.2f}个归一化屏幕单位。"
            )

    @staticmethod
    def _validate_targeted_swipe_absence_contract(
        element: UIElement,
        absence: Any,
    ) -> None:
        if not isinstance(absence, Mapping) or set(absence) != {
            "element_id",
            "meaning",
            "role",
            "label",
        }:
            raise UniversalActionError(
                "元素绑定滑动必须精确声明同一目标的 element_absent 后置条件。"
            )
        expected = {
            "element_id": element.element_id,
            "meaning": element.meaning,
            "role": element.role,
            "label": element.label,
        }
        if dict(absence) != expected:
            raise UniversalActionError(
                "元素绑定滑动的消失目标与当前可信元素不一致。"
            )

    @classmethod
    def _targeted_swipe_path(
        cls,
        element: UIElement,
        direction: str,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        left, top, right, bottom = element.bounds
        width = right - left
        height = bottom - top
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        if direction == "up":
            start = (center_x, top + height * 0.75)
            end = (center_x, TARGETED_SWIPE_EDGE_MARGIN)
        elif direction == "down":
            start = (center_x, top + height * 0.25)
            end = (center_x, 1.0 - TARGETED_SWIPE_EDGE_MARGIN)
        elif direction == "left":
            start = (left + width * 0.75, center_y)
            end = (TARGETED_SWIPE_EDGE_MARGIN, center_y)
        elif direction == "right":
            start = (left + width * 0.25, center_y)
            end = (1.0 - TARGETED_SWIPE_EDGE_MARGIN, center_y)
        else:
            raise UniversalActionError(f"不支持的元素滑动方向：{direction}")
        cls._validate_gesture_point(start, label="元素滑动起点")
        cls._validate_gesture_point(end, label="元素滑动终点")
        delta_x = end[0] - start[0]
        delta_y = end[1] - start[1]
        direction_matches = {
            "up": delta_y < 0 and abs(delta_y) > abs(delta_x),
            "down": delta_y > 0 and abs(delta_y) > abs(delta_x),
            "left": delta_x < 0 and abs(delta_x) > abs(delta_y),
            "right": delta_x > 0 and abs(delta_x) > abs(delta_y),
        }[direction]
        distance = math.dist(start, end)
        if not direction_matches:
            raise UniversalActionError("元素边界无法形成指定方向的滑动轨迹。")
        if not MIN_DRAG_DISTANCE <= distance <= MAX_DRAG_DISTANCE:
            raise UniversalActionError(
                "元素滑动轨迹距离必须在"
                f"{MIN_DRAG_DISTANCE:.2f}～{MAX_DRAG_DISTANCE:.2f}之间。"
            )
        return start, end

    @staticmethod
    def _element_identity_surface_is_continuous(
        before: UIScene,
        after: UIScene,
    ) -> bool:
        """Return whether observation-local element identity may carry over.

        ``element_id`` is minted independently for every observation.  It can
        therefore help identify a surviving element only while the typed
        surface itself is continuous; it is never a global identity across a
        navigation transition.
        """

        if scene_surface_kind(before) != scene_surface_kind(after):
            return False
        before_app = before.foreground_app_id.strip().casefold()
        after_app = after.foreground_app_id.strip().casefold()
        if (
            before_app not in {"", "unknown"}
            and after_app not in {"", "unknown"}
            and before_app != after_app
        ):
            return False
        before_screen = before.screen_id.strip().casefold()
        after_screen = after.screen_id.strip().casefold()
        if (
            before_screen not in {"", "unknown"}
            and after_screen not in {"", "unknown"}
            and before_screen != after_screen
        ):
            return False
        return True

    @staticmethod
    def _element_regions_stably_overlap(
        before_bounds: tuple[float, float, float, float],
        after_bounds: tuple[float, float, float, float],
    ) -> bool:
        left = max(before_bounds[0], after_bounds[0])
        top = max(before_bounds[1], after_bounds[1])
        right = min(before_bounds[2], after_bounds[2])
        bottom = min(before_bounds[3], after_bounds[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        before_area = max(0.0, before_bounds[2] - before_bounds[0]) * max(
            0.0, before_bounds[3] - before_bounds[1]
        )
        after_area = max(0.0, after_bounds[2] - after_bounds[0]) * max(
            0.0, after_bounds[3] - after_bounds[1]
        )
        smaller = min(before_area, after_area)
        return intersection > 0 and smaller > 0 and intersection / smaller >= 0.60

    @classmethod
    def _is_same_absence_target(
        cls,
        before_target: UIElement,
        after_element: UIElement,
        *,
        surface_is_continuous: bool,
    ) -> bool:
        before_label = before_target.label.strip().casefold()
        after_label = after_element.label.strip().casefold()
        before_role = before_target.role.strip().casefold()
        after_role = after_element.role.strip().casefold()
        before_meaning = before_target.meaning.strip().casefold()
        after_meaning = after_element.meaning.strip().casefold()
        label_matches = bool(before_label) and before_label == after_label
        exact_semantics = (
            bool(before_meaning)
            and before_role == after_role
            and before_meaning == after_meaning
            and (
                label_matches
                or not before_label
                or not after_label
            )
        )
        if exact_semantics:
            return True
        if not surface_is_continuous:
            return False
        spatially_continuous = cls._element_regions_stably_overlap(
            before_target.bounds,
            after_element.bounds,
        )
        return spatially_continuous and (
            after_element.element_id == before_target.element_id
            or label_matches
        )

    def _verify_expected_element_absent(
        self,
        resolved: ResolvedSemanticAction,
        before: UIScene,
        after: UIScene,
    ) -> None:
        absence = resolved.expected_effect.get("element_absent")
        target_id = str(resolved.target_element_id or "").strip()
        if resolved.kind != "swipe" or not target_id:
            raise UniversalActionError(
                "element_absent 结果没有绑定元素滑动动作。"
            )
        try:
            before_target = before.get_element(
                target_id,
                min_confidence=self.min_confidence,
            )
        except UISceneError as exc:
            raise UniversalActionError(
                f"元素滑动前目标证据无效：{exc}"
            ) from exc
        self._validate_targeted_swipe_absence_contract(
            before_target,
            absence,
        )
        surface_is_continuous = self._element_identity_surface_is_continuous(
            before,
            after,
        )
        still_visible = tuple(
            element
            for element in after.elements
            if float(element.confidence) >= self.min_confidence
            and element.states.get("visible") is not False
            and self._is_same_absence_target(
                before_target,
                element,
                surface_is_continuous=surface_is_continuous,
            )
        )
        if still_visible:
            raise UniversalActionError(
                "元素滑动后同一目标仍然可见，不能判定已划掉。"
            )

    @staticmethod
    def _has_structured_postcondition(
        expected: dict[str, Any],
        before: UIScene,
    ) -> bool:
        if any(
            expected.get(key) is True
            for key in ("scene_changed", "content_changed", "current_video_changed")
        ):
            return True
        expected_app = str(expected.get("app_id") or "").strip()
        if expected_app and expected_app != before.foreground_app_id:
            return True
        expected_screen = str(expected.get("screen_id") or "").strip()
        if expected_screen and expected_screen != before.screen_id:
            return True
        element_state = expected.get("element_state")
        element_absent = expected.get("element_absent")
        if isinstance(element_absent, Mapping):
            return bool(str(element_absent.get("element_id") or "").strip())
        return bool(
            isinstance(element_state, dict)
            and str(element_state.get("meaning") or "").strip()
            and isinstance(element_state.get("states"), dict)
            and element_state["states"]
        )

    @classmethod
    def _require_visual_postcondition(
        cls,
        kind: str,
        expected: dict[str, Any],
        before: UIScene,
    ) -> None:
        if expected.get("allow_unchanged") is True:
            raise UniversalActionError(f"{kind} 禁止声明 allow_unchanged。")
        if not cls._has_structured_postcondition(expected, before):
            raise UniversalActionError(
                f"{kind} 必须声明可由动作后新画面验证的结构化预期。"
            )

    def _verify_drag_result(
        self,
        resolved: ResolvedSemanticAction,
        before: UIScene,
        after: UIScene,
    ) -> None:
        source_id = str(resolved.target_element_id or "").strip()
        destination_id = str(resolved.destination_element_id or "").strip()
        if not source_id or not destination_id:
            raise UniversalActionError("拖动结果缺少起点或终点元素身份。")
        try:
            source = before.get_element(source_id, min_confidence=self.min_confidence)
            destination = before.get_element(
                destination_id,
                min_confidence=self.min_confidence,
            )
        except UISceneError as exc:
            raise UniversalActionError(f"拖动前端点证据无效：{exc}") from exc

        source_container_error = compact_drag_source_container_error(before, source)
        if source_container_error:
            raise UniversalActionError(source_container_error)
        self._validate_gesture_point(source.center, label="拖动起点")
        self._validate_gesture_point(destination.center, label="拖动终点")

        distance = math.dist(source.center, destination.center)
        if not MIN_DRAG_DISTANCE <= distance <= MAX_DRAG_DISTANCE:
            raise UniversalActionError("拖动路径距离超出安全范围。")
        if (
            resolved.path_distance is None
            or abs(float(resolved.path_distance) - distance) > 1e-6
        ):
            raise UniversalActionError("拖动路径距离与动作前端点不一致。")
        if resolved.hold_seconds != DRAG_DURATION_SECONDS:
            raise UniversalActionError("拖动执行时长不是控制器固定的0.8秒。")

        exact_id = tuple(
            element
            for element in after.elements
            if element.element_id == source_id
            and element.role == source.role
            and float(element.confidence) >= self.min_confidence
            and element.states.get("visible") is not False
        )
        if exact_id:
            candidates = exact_id
        else:
            candidates = tuple(
                element
                for element in after.elements
                if element.role == source.role
                and float(element.confidence) >= self.min_confidence
                and element.states.get("visible") is not False
                and element.meaning.casefold() == source.meaning.casefold()
                and element.label.casefold() == source.label.casefold()
            )
        if len(candidates) == 1:
            moved = math.dist(source.center, candidates[0].center)
            remaining = math.dist(candidates[0].center, destination.center)
            required_improvement = max(0.03, distance * 0.20)
            if (
                moved >= MIN_DRAG_RESULT_DISPLACEMENT
                and remaining <= distance - required_improvement
            ):
                return

        expected = resolved.expected_effect
        has_alternative_proof = bool(
            self._element_state_transition_expected(expected, before)
            or (
                str(expected.get("app_id") or "").strip()
                and str(expected.get("app_id") or "").strip()
                != before.foreground_app_id
            )
            or (
                str(expected.get("screen_id") or "").strip()
                and str(expected.get("screen_id") or "").strip()
                != before.screen_id
            )
        )
        if not has_alternative_proof:
            raise UniversalActionError(
                "拖动后缺少源元素向终点显著移动或等价结构化状态证据。"
            )

    def _verify_long_press_result(
        self,
        resolved: ResolvedSemanticAction,
        before: UIScene,
        after: UIScene,
    ) -> None:
        if set(after.overlays) - set(before.overlays):
            return
        target_id = str(resolved.target_element_id or "").strip()
        try:
            before_target = before.get_element(
                target_id,
                min_confidence=self.min_confidence,
            )
        except UISceneError as exc:
            raise UniversalActionError(f"长按前目标证据无效：{exc}") from exc
        after_targets = tuple(
            element
            for element in after.elements
            if element.element_id == target_id
            and element.role == before_target.role
            and float(element.confidence) >= self.min_confidence
            and element.states.get("visible") is not False
        )
        if len(after_targets) == 1 and after_targets[0].states != before_target.states:
            return
        result_markers = (
            "verification",
            "status",
            "result",
            "outcome",
            "feedback",
            "验证",
            "状态",
            "结果",
            "反馈",
        )
        new_structured_results = tuple(
            element
            for element in after.elements
            if element.role in {"text", "button", "icon"}
            and element.states.get("goal_relevant") is True
            and element.states.get("fully_visible") is True
            and float(element.confidence) >= self.min_confidence
            and bool(str(element.label or "").strip())
            and bool(element.evidence)
            and any(
                marker in str(element.meaning or "").casefold()
                for marker in result_markers
            )
            and not any(
                prior.role == element.role
                and prior.meaning == element.meaning
                and prior.label == element.label
                for prior in before.elements
            )
        )
        if len(new_structured_results) == 1:
            return
        expected = resolved.expected_effect
        if self._element_state_transition_expected(expected, before):
            return
        if (
            str(expected.get("app_id") or "").strip()
            and str(expected.get("app_id") or "").strip() != before.foreground_app_id
        ) or (
            str(expected.get("screen_id") or "").strip()
            and str(expected.get("screen_id") or "").strip() != before.screen_id
        ):
            return
        raise UniversalActionError(
            "长按后缺少新增弹层、目标状态变化或等价结构化结果证据。"
        )

    def _verify_formal_transition(
        self,
        resolved: ResolvedSemanticAction,
        before: UIScene,
        after: UIScene,
    ) -> None:
        transition = resolved.formal_transition
        expectations = transition.get("expectations") if isinstance(transition, dict) else None
        if not isinstance(expectations, list) or not expectations:
            raise UniversalActionError("正式候选缺少 typed transition expectations。")
        for expectation in expectations:
            if not isinstance(expectation, dict):
                raise UniversalActionError("typed transition expectation 格式无效。")
            predicate = str(expectation.get("predicate") or "")
            operator = str(expectation.get("operator") or "")
            value = expectation.get("value")
            if predicate == "surface.kind" and operator == "equals":
                actual = scene_surface_kind(after)
                if actual != value:
                    raise UniversalActionError("typed surface.kind 后置状态未满足。")
            elif predicate == "surface.overlay_present" and operator == "equals":
                if bool(after.overlays) is not bool(value):
                    raise UniversalActionError("typed overlay 后置状态未满足。")
            elif predicate == "element.exists" and operator == "absent":
                self._verify_expected_element_absent(resolved, before, after)
            elif predicate == "system_ui.navigation_bar_visible" and operator == "equals":
                if after.system_ui.navigation_bar_visible is not value:
                    raise UniversalActionError("typed system_ui 后置状态未满足。")
            elif predicate == "element.state.value" and operator == "equals":
                expected_transition_value = (
                    resolved.prior_input_value
                    if (
                        resolved.kind == "input_verified_text"
                        and resolved.input_method == "chinese_pinyin"
                    )
                    else resolved.expected_input_value
                )
                if expected_transition_value != value and resolved.text != value:
                    raise UniversalActionError("typed input value 与已验证事务不一致。")
            elif (
                predicate == "element.state.ime_preedit_text"
                and operator == "equals"
            ):
                if (
                    resolved.kind != "input_verified_text"
                    or resolved.input_method != "chinese_pinyin"
                    or resolved.input_pinyin != value
                ):
                    raise UniversalActionError(
                        "typed 拼音组合状态与已验证中文事务不一致。"
                    )
            elif (
                predicate == "element.state.ime_preedit_text"
                and operator == "absent"
            ):
                if resolved.kind != "clear_verified_text":
                    raise UniversalActionError(
                        "typed 预编辑清空后置状态未绑定清空动作。"
                    )
            elif (
                predicate == "element.state.ime_exact_candidate_text"
                and operator == "equals"
            ):
                if (
                    resolved.kind != "input_verified_text"
                    or resolved.input_method != "chinese_pinyin"
                    or resolved.input_fragment != value
                ):
                    raise UniversalActionError(
                        "typed 中文候选状态与已验证中文事务不一致。"
                    )
            elif predicate in {
                "element.state.keyboard_layout",
                "element.state.keyboard_input_mode",
                "element.state.keyboard_case_mode",
            } and operator == "equals":
                state_key = predicate.removeprefix("element.state.")
                expected_element = resolved.expected_effect.get("element_state")
                expected_states = (
                    expected_element.get("states")
                    if isinstance(expected_element, dict)
                    else None
                )
                input_element_id = str(resolved.input_element_id or "").strip()
                if (
                    resolved.kind != "tap_semantic"
                    or not input_element_id
                    or not isinstance(value, str)
                    or not value
                    or not isinstance(expected_states, dict)
                    or expected_states.get(state_key) != value
                    or expected_states.get("value")
                    != resolved.expected_input_value
                ):
                    raise UniversalActionError(
                        f"typed {state_key} 与已验证输入辅助动作不一致。"
                    )
                matches = tuple(
                    item
                    for item in after.elements
                    if item.element_id == input_element_id
                    and item.role == "input"
                    and item.states.get(state_key) == value
                    and item.states.get("value")
                    == resolved.expected_input_value
                    and float(item.confidence) >= MIN_TARGET_CONFIDENCE
                    and item.states.get("visible") is not False
                )
                if len(matches) != 1:
                    raise UniversalActionError(
                        f"typed {state_key} 后置状态未满足。"
                    )
            elif predicate == "effect.applied" and operator == "equals":
                if value is not True or before.fingerprint == after.fingerprint:
                    raise UniversalActionError("typed effect receipt 缺少动作后变化证据。")
            elif predicate in {
                "surface.active_ref",
                "surface.focused_entity_ref",
            } and operator == "equals":
                if before.fingerprint == after.fingerprint:
                    raise UniversalActionError("typed surface 目标没有产生新观察。")
            elif predicate in {
                "surface.navigation_depth",
                "surface.viewport",
                "observation.changed",
                "scene.changed",
                "element.state.interaction_result",
                "element.state.location_relation",
            } and operator == "changed":
                if before.fingerprint == after.fingerprint:
                    raise UniversalActionError("typed changed 后置状态未满足。")
            elif predicate in {
                "element.state.focused",
            } and operator == "equals":
                target_id = str(resolved.target_element_id or "")
                matches = [item for item in after.elements if item.element_id == target_id]
                if len(matches) == 1 and matches[0].states.get("focused") is value:
                    continue
                focus_only_sources = tuple(
                    item
                    for item in before.elements
                    if item.element_id == target_id
                    and item.role == "input"
                    and item.states.get("focus_only_input_surface") is True
                )
                audited_focus_matches = tuple(
                    item
                    for item in after.elements
                    if item.role == "input"
                    and item.meaning == "application_text_input"
                    and item.element_id.startswith("local_audited_")
                    and item.states.get("focused") is value
                    and item.states.get("fully_visible") is True
                    and item.states.get("primary_input_geometry_verified") is True
                    and item.states.get("geometry_audit_source")
                    == "input_structure_audit"
                    and str(item.states.get("input_field_id") or "").strip()
                    not in {"", "unknown"}
                    and float(item.confidence) >= self.min_confidence
                )
                focus_only_transition_verified = bool(
                    value is True
                    and len(focus_only_sources) == 1
                    and len(audited_focus_matches) == 1
                    and input_app_identity_compatible(
                        before.foreground_app_id,
                        after.foreground_app_id,
                    )
                    and input_screen_identity_compatible(
                        before.screen_id,
                        after.screen_id,
                    )
                )
                if not focus_only_transition_verified:
                    raise UniversalActionError("typed focused 后置状态未满足。")
            elif predicate == "input_field.focused" and operator == "equals":
                executed_targets = [
                    item for item in before.elements
                    if item.element_id == resolved.target_element_id
                    and item.meaning == "input_next_field_key"
                ]
                target_label = (
                    str(executed_targets[0].states.get("target_input_field_label") or "").strip()
                    if len(executed_targets) == 1
                    else ""
                )
                matches = [
                    item for item in after.elements
                    if item.role == "input"
                    and item.states.get("input_field_id") == expectation.get("subject_ref")
                    and item.states.get("input_field_label") == target_label
                    and item.states.get("focused") is value
                    and float(item.confidence) >= MIN_TARGET_CONFIDENCE
                ]
                if value is not True or len(matches) != 1:
                    raise UniversalActionError(
                        "typed目标字段聚焦后置状态未满足。"
                    )
            elif predicate == "element.state.location_relation" and operator == "equals":
                if (
                    resolved.kind != "drag"
                    or not resolved.target_element_id
                    or not resolved.destination_element_id
                    or not isinstance(value, str)
                    or not value
                ):
                    raise UniversalActionError("typed drag relation 与已解析动作不一致。")
                # The concrete spatial outcome is verified by
                # _verify_drag_result immediately after this contract check.
            else:
                raise UniversalActionError(
                    f"尚未实现的 typed transition expectation：{predicate}/{operator}"
                )

    def _element_state_transition_expected(
        self,
        expected: dict[str, Any],
        before: UIScene,
    ) -> bool:
        element_state = expected.get("element_state")
        if not isinstance(element_state, dict):
            return False
        meaning = str(element_state.get("meaning") or "").strip()
        states = element_state.get("states")
        if not meaning or not isinstance(states, dict) or not states:
            return False
        matches = tuple(
            element
            for element in before.elements
            if element.meaning == meaning
            and float(element.confidence) >= self.min_confidence
            and element.states.get("visible") is not False
            and all(element.states.get(key) == value for key, value in states.items())
        )
        return not matches

    def _verify_long_press_contract(
        self,
        resolved: ResolvedSemanticAction,
        before: UIScene,
    ) -> None:
        target_id = str(resolved.target_element_id or "").strip()
        if not target_id or resolved.normalized_point is None:
            raise UniversalActionError("长按结果缺少目标元素身份或落点。")
        if (
            resolved.hold_seconds is None
            or not 0.5 <= float(resolved.hold_seconds) <= 2.0
        ):
            raise UniversalActionError("长按执行时长必须在0.5～2.0秒之间。")
        try:
            target = before.get_element(target_id, min_confidence=self.min_confidence)
        except UISceneError as exc:
            raise UniversalActionError(f"长按前目标证据无效：{exc}") from exc
        if target.role == "container":
            raise UniversalActionError("页面容器不是可长按控件。")
        self._validate_gesture_point(target.center, label="长按落点")
        if math.dist(resolved.normalized_point, target.center) > 1e-6:
            raise UniversalActionError("长按落点与动作前目标中心不一致。")

    def _verify_exact_input_value(
        self,
        resolved: ResolvedSemanticAction,
        before: UIScene,
        after: UIScene,
    ) -> None:
        expected = (
            resolved.expected_input_value
            if resolved.kind in {"input_verified_text", "press_enter"}
            else resolved.text
        )
        target_id = str(
            resolved.input_element_id or resolved.target_element_id or ""
        ).strip()
        if expected is None or not target_id:
            raise UniversalActionError("输入动作缺少精确文字或目标输入框身份。")
        if resolved.kind == "input_verified_text" and (
            not expected or not resolved.input_fragment or not resolved.input_method
        ):
            raise UniversalActionError("输入动作缺少精确文字或目标输入框身份。")
        if resolved.kind == "press_enter" and (
            resolved.prior_input_value is None
            or expected != resolved.prior_input_value + "\n"
        ):
            raise UniversalActionError("换行动作缺少精确前缀或 newline 后置值。")
        if resolved.kind == "clear_verified_text":
            if expected != "" or resolved.delete_count is None:
                raise UniversalActionError("清空动作缺少空值或精确退格次数。")
        try:
            before_input = before.get_element(
                target_id,
                min_confidence=self.min_confidence,
            )
        except UISceneError as exc:
            raise UniversalActionError(f"输入前目标证据无效：{exc}") from exc
        if before_input.role != "input":
            raise UniversalActionError("输入前目标不是 input 元素。")
        if (
            not resolved.formal_candidate_id
            and before_input.states.get("goal_relevant") is not True
        ):
            raise UniversalActionError("输入前目标与当前目标缺少可信关联。")
        typed_field_id = str(
            before_input.states.get("input_field_id") or ""
        ).strip()
        if typed_field_id and typed_field_id != "unknown":
            candidates = tuple(
                element
                for element in after.elements
                if element.role == "input"
                and float(element.confidence) >= self.min_confidence
                and element.states.get("visible") is not False
                and element.meaning == before_input.meaning == "application_text_input"
                and str(element.states.get("input_field_id") or "").strip()
                == typed_field_id
            )
        else:
            exact_id = tuple(
                element
                for element in after.elements
                if element.element_id == target_id
                and element.role == "input"
                and float(element.confidence) >= self.min_confidence
                and element.states.get("visible") is not False
            )
            if exact_id:
                candidates = exact_id
            else:
                semantic_candidates = tuple(
                    element
                    for element in after.elements
                    if element.role == "input"
                    and float(element.confidence) >= self.min_confidence
                    and element.states.get("visible") is not False
                    and element.meaning.casefold() == before_input.meaning.casefold()
                    and element.label.casefold() == before_input.label.casefold()
                )
                if len(semantic_candidates) == 1:
                    candidates = semantic_candidates
                else:
                    candidates = tuple(
                        element
                        for element in after.elements
                        if element.role == "input"
                        and float(element.confidence) >= self.min_confidence
                        and element.states.get("visible") is not False
                        and self._input_regions_stably_overlap(
                            before_input.bounds,
                            element.bounds,
                        )
                    )
        if len(candidates) != 1:
            raise UniversalActionError("动作后无法唯一绑定原目标输入框。")
        states = candidates[0].states
        if "value" not in states or not isinstance(states["value"], str):
            raise UniversalActionError("动作后缺少输入框 states.value 精确文字证据。")
        actual = states["value"]
        if resolved.input_method == "chinese_pinyin":
            if actual != resolved.prior_input_value:
                raise UniversalActionError(
                    "拼音键入后应用输入值在候选确认前已意外变化。"
                )
            if states.get("ime_preedit_text") != resolved.input_pinyin:
                raise UniversalActionError("动作后缺少逐字一致的拼音组合证据。")
            if states.get("ime_exact_candidate_text") != resolved.input_fragment:
                raise UniversalActionError("动作后缺少唯一逐字一致的中文候选。")
        elif (
            resolved.kind == "input_verified_text"
            and resolved.input_method == "direct_latin"
            and actual != expected
        ):
            # Some system keyboards keep an exact Latin key sequence in an
            # IME composition buffer until its visible identical candidate is
            # selected.  This is verified progress, not a committed App value.
            # Admit it only when the same typed field remains empty at the
            # prior value and one locally audited candidate proves the exact
            # fragment and resulting value.  The next physical action must
            # still select that candidate and verify the committed App value.
            if not self._is_exact_direct_latin_preedit_transition(
                resolved,
                after,
                candidates[0],
            ):
                raise UniversalActionError(
                    f"动作后输入框文字不匹配：实际 {actual!r}，预期 {expected!r}。"
                )
        elif actual != expected:
            raise UniversalActionError(
                f"动作后输入框文字不匹配：实际 {actual!r}，预期 {expected!r}。"
            )
        if (
            resolved.kind == "clear_verified_text"
            and before_input.states.get("ime_preedit_text")
        ):
            if states.get("ime_preedit_text") not in (None, ""):
                raise UniversalActionError("清空动作后输入法预编辑文字仍未清除。")
            if states.get("focused") is not True:
                raise UniversalActionError("清空动作后原typed输入框不再聚焦。")
        after_input = candidates[0]
        if not self._input_scene_identity_is_stable(
            before,
            after,
            before_input,
            after_input,
        ):
            raise UniversalActionError("输入动作后 App 或页面身份发生变化。")

    def _is_exact_direct_latin_preedit_transition(
        self,
        resolved: ResolvedSemanticAction,
        after: UIScene,
        after_input: UIElement,
    ) -> bool:
        expected = resolved.expected_input_value
        fragment = resolved.input_fragment
        prior = resolved.prior_input_value
        states = after_input.states
        exact_candidates = tuple(
            element
            for element in after.elements
            if element.meaning == "ime_exact_candidate"
            and element.label == fragment
            and float(element.confidence) >= self.min_confidence
            and element.states.get("goal_relevant") is True
            and element.states.get("fully_visible") is True
            and element.states.get("ime_candidate") is True
            and element.states.get("input_element_id") == after_input.element_id
            and element.states.get("prior_input_value") == prior
            and element.states.get("expected_input_value") == expected
            and element.states.get("pinyin") == fragment
        )
        return bool(
            resolved.kind == "input_verified_text"
            and resolved.input_method == "direct_latin"
            and isinstance(prior, str)
            and isinstance(fragment, str)
            and fragment
            and expected == prior + fragment
            and states.get("value") == prior
            and states.get("ime_preedit_text") == fragment
            and states.get("ime_exact_candidate_text") == fragment
            and len(exact_candidates) == 1
        )

    @classmethod
    def _input_scene_identity_is_stable(
        cls,
        before: UIScene,
        after: UIScene,
        before_input: UIElement,
        after_input: UIElement,
    ) -> bool:
        if (
            before.foreground_app_id == after.foreground_app_id
            and before.screen_id == after.screen_id
        ):
            return True

        app_identity_compatible = input_app_identity_compatible(
            before.foreground_app_id,
            after.foreground_app_id,
        )
        before_states = before_input.states
        after_states = after_input.states
        typed_field_id = str(before_states.get("input_field_id") or "").strip()
        before_field_label = str(
            before_states.get("input_field_label") or ""
        ).strip()
        after_field_label = str(
            after_states.get("input_field_label") or ""
        ).strip()
        typed_field_identity = bool(
            typed_field_id
            and typed_field_id != "unknown"
            and str(after_states.get("input_field_id") or "").strip()
            == typed_field_id
            and before_input.meaning == after_input.meaning == "application_text_input"
            and before_states.get("fully_visible") is not False
            and after_states.get("fully_visible") is not False
            and isinstance(before_states.get("input_multiline"), bool)
            and after_states.get("input_multiline")
            == before_states.get("input_multiline")
            and not (
                before_field_label
                and after_field_label
                and before_field_label != after_field_label
            )
        )
        screen_identity_compatible = input_screen_identity_compatible(
            before.screen_id,
            after.screen_id,
        )
        if not screen_identity_compatible and not typed_field_identity:
            return False
        if not app_identity_compatible:
            if (
                input_app_identity_is_concrete_package(
                    before.foreground_app_id
                )
                or input_app_identity_is_concrete_package(
                    after.foreground_app_id
                )
            ):
                return False
            before_family = input_screen_identity_family(before.screen_id)
            after_family = input_screen_identity_family(after.screen_id)
            if (
                (not before_family or before_family != after_family)
                and not typed_field_identity
            ):
                return False
        if not (
            cls._input_regions_stably_overlap(
                before_input.bounds,
                after_input.bounds,
            )
            or typed_field_identity
        ):
            return False
        if before_states.get("focused") is not True or after_states.get("focused") is not True:
            return False
        before_layout = before_states.get("keyboard_layout")
        after_layout = after_states.get("keyboard_layout")
        if (
            before_layout in {None, "unknown"}
            or after_layout in {None, "unknown"}
            or before_layout != after_layout
        ):
            return False
        before_mode = before_states.get("keyboard_input_mode")
        after_mode = after_states.get("keyboard_input_mode")
        preedit_clear_transition = bool(
            typed_field_identity
            and before_states.get("value") == ""
            and isinstance(before_states.get("ime_preedit_text"), str)
            and bool(before_states.get("ime_preedit_text"))
            and after_states.get("value") == ""
            and after_states.get("ime_preedit_text") in {None, ""}
        )
        preedit_start_transition = bool(
            typed_field_identity
            and isinstance(before_states.get("value"), str)
            and after_states.get("value") == before_states.get("value")
            and before_states.get("ime_preedit_text") in {None, ""}
            and isinstance(after_states.get("ime_preedit_text"), str)
            and bool(after_states.get("ime_preedit_text"))
            and after_states.get("ime_exact_candidate_text")
            == after_states.get("ime_preedit_text")
        )
        if (
            before_mode in {None, "unknown"}
            or after_mode in {None, "unknown"}
            or (
                before_mode != after_mode
                and not preedit_clear_transition
                and not preedit_start_transition
            )
        ):
            return False
        return (
            before.camera_alignment.camera_layout_orientation
            == after.camera_alignment.camera_layout_orientation
            and before.camera_alignment.phone_content_rotation
            == after.camera_alignment.phone_content_rotation
        )

    @staticmethod
    def _input_regions_stably_overlap(
        before_bounds: tuple[float, float, float, float],
        after_bounds: tuple[float, float, float, float],
    ) -> bool:
        left = max(before_bounds[0], after_bounds[0])
        top = max(before_bounds[1], after_bounds[1])
        right = min(before_bounds[2], after_bounds[2])
        bottom = min(before_bounds[3], after_bounds[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        before_area = max(0.0, before_bounds[2] - before_bounds[0]) * max(
            0.0, before_bounds[3] - before_bounds[1]
        )
        after_area = max(0.0, after_bounds[2] - after_bounds[0]) * max(
            0.0, after_bounds[3] - after_bounds[1]
        )
        smaller = min(before_area, after_area)
        return intersection > 0 and smaller > 0 and intersection / smaller >= 0.60

    @classmethod
    def scenes_semantically_equivalent(
        cls,
        before: UIScene,
        after: UIScene,
    ) -> bool:
        """Ignore camera noise and model box jitter when comparing scenes."""

        def freeze(value: Any) -> Any:
            if isinstance(value, dict):
                return tuple(
                    sorted((str(key), freeze(item)) for key, item in value.items())
                )
            if isinstance(value, (list, tuple)):
                return tuple(freeze(item) for item in value)
            return value

        def signature(scene: UIScene) -> tuple[Any, ...]:
            elements = tuple(
                sorted(
                    (
                        element.role.casefold(),
                        element.label.casefold(),
                        # Visible text plus role/states is stronger cross-frame
                        # identity than model-authored meaning wording.  For an
                        # unlabeled icon we still need meaning to identify it.
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
            return (
                scene.foreground_app_id.casefold(),
                scene.screen_id.casefold(),
                tuple(sorted(item.casefold() for item in scene.overlays)),
                elements,
            )

        before.validate()
        after.validate()
        return signature(before) == signature(after)

    def transition_evidence_from_verified_action(
        self,
        resolved: ResolvedSemanticAction,
        before: UIScene,
        after: UIScene,
    ) -> tuple[str, ...]:
        """Describe a transition already accepted by ``verify_after_action``.

        These facts describe only the verified transition. DeepSeek remains
        the sole author of task completion.  This method deliberately does
        not verify again; the physical adapter calls the verifier exactly once
        before asking for this evidence projection.
        """

        expected = resolved.expected_effect
        evidence: list[str] = []

        scene_change_requested = any(
            expected.get(key) is True
            for key in ("scene_changed", "content_changed", "current_video_changed")
        )
        if scene_change_requested:
            if not before.fingerprint or not after.fingerprint:
                return ()
            if before.fingerprint == after.fingerprint:
                return ()
            evidence.append(
                "控制器确认动作前后场景指纹发生变化："
                f"{before.fingerprint} -> {after.fingerprint}"
            )

        expected_app = str(expected.get("app_id") or "").strip()
        if expected_app:
            if after.foreground_app_id != expected_app:
                return ()
            evidence.append(f"控制器确认前台 App 为 {expected_app}")

        expected_screen = str(expected.get("screen_id") or "").strip()
        if expected_screen:
            if after.screen_id != expected_screen:
                return ()
            evidence.append(f"控制器确认页面为 {expected_screen}")

        element_state = expected.get("element_state")
        if element_state is not None:
            if not isinstance(element_state, dict):
                return ()
            meaning = str(element_state.get("meaning") or "").strip()
            states = dict(element_state.get("states") or {})
            try:
                element = after.resolve_unique(
                    meaning=meaning,
                    states=states,
                    min_confidence=self.min_confidence,
                )
            except UISceneError:
                return ()
            evidence.append(
                "控制器确认目标元素状态："
                f"{element.meaning} {element.states}"
            )

        element_absent = expected.get("element_absent")
        if isinstance(element_absent, Mapping):
            self._verify_expected_element_absent(resolved, before, after)
            identity = (
                str(element_absent.get("label") or "").strip()
                or str(element_absent.get("meaning") or "").strip()
                or str(element_absent.get("element_id") or "").strip()
            )
            evidence.append(f"控制器确认目标元素已消失：{identity}")

        return tuple(evidence)

    def _resolve_target(
        self,
        action: SemanticAction,
        scene: UIScene,
        *,
        prefix: str = "",
        required_role: str | None = None,
    ) -> UIElement:
        target = str(action.params.get(f"{prefix}target") or "").strip()
        element_id = str(action.params.get(f"{prefix}element_id") or "").strip()
        if not target and not element_id:
            raise UniversalActionError(
                f"{action.action} 缺少 {prefix}target 或 {prefix}element_id。"
            )
        role = str(action.params.get(f"{prefix}role") or "").strip() or None
        if required_role is not None:
            if role is not None and role != required_role:
                raise UniversalActionError(f"动作目标角色必须为 {required_role}。")
            role = required_role
        label = str(action.params.get(f"{prefix}label") or "").strip() or None
        states = action.params.get(f"{prefix}states") or {}
        if not isinstance(states, dict):
            raise UniversalActionError(f"{action.action}.{prefix}states 格式无效。")
        try:
            if element_id:
                element = scene.get_element(
                    element_id,
                    min_confidence=self.min_confidence,
                )
                if target and target.casefold() not in {
                    element.meaning.casefold(),
                    element.label.casefold(),
                }:
                    raise UISceneError(
                        f"元素 {element_id} 的语义与目标不一致：{target}"
                    )
                if role and element.role != role:
                    raise UISceneError(
                        f"元素 {element_id} 的角色与目标不一致：{role}"
                    )
                if label and element.label.casefold() != label.casefold():
                    raise UISceneError(
                        f"元素 {element_id} 的文字与目标不一致：{label}"
                    )
                if any(element.states.get(key) != value for key, value in states.items()):
                    raise UISceneError(f"元素 {element_id} 的状态与目标不一致。")
                return element
            return scene.resolve_unique(
                meaning=target,
                label=label,
                role=role,
                states=states,
                min_confidence=self.min_confidence,
            )
        except UISceneError as exc:
            raise UniversalActionError(str(exc)) from exc

    @staticmethod
    def _point_action(
        action: SemanticAction,
        element: UIElement,
        expected_effect: dict[str, Any],
        before_fingerprint: str,
        *,
        scene: UIScene | None = None,
        local_point_grounding: LocalPointGrounding | None = None,
    ) -> ResolvedSemanticAction:
        if element.role == "container":
            raise UniversalActionError("页面容器不是可点击控件，禁止执行点击。")
        normalized_point = element.center
        proposed_point: tuple[float, float] | None = None
        grounding_payload: dict[str, Any] | None = None
        if local_point_grounding is not None:
            if scene is None:
                raise UniversalActionError("本地落点证据缺少当前场景绑定。")
            local_point_grounding.validate_for(scene, element)
            proposed_point = element.center
            normalized_point = local_point_grounding.grounded_point
            grounding_payload = local_point_grounding.to_dict()
            if action.action in {"double_tap", "long_press"}:
                UniversalActionController._validate_gesture_point(
                    normalized_point,
                    label="本地修正手势落点",
                )
        return ResolvedSemanticAction(
            node_id=action.node_id,
            kind=action.action,
            normalized_point=normalized_point,
            target_element_id=element.element_id,
            before_fingerprint=before_fingerprint,
            expected_effect=expected_effect,
            formal_candidate_id=str(
                action.params.get("formal_candidate_id") or ""
            ).strip(),
            formal_transition=dict(action.params.get("formal_transition") or {}),
            proposed_normalized_point=proposed_point,
            point_grounding=grounding_payload,
        )
