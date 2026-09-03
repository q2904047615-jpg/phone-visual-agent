"""Pure domain service that validates and resolves one current-frame action.

The controller is deliberately not a second goal or visual authority.  Qwen
selects one action from the current screenshot.  This service only proves that
the selected action is executable on that same scene, resolves its geometry,
and checks receipts that have an exact local contract (trusted launch, system
navigation, and verified text transport).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import math
from typing import Any

from agent.domain.semantic_action import SemanticAction
from agent.domain.text_input_utils import editable_character_count, normalize_user_text
from agent.domain.ui_scene import UIElement, UIScene, UISceneError, scene_matches_app_identity
from agent.domain.validation import DataclassWire, NormalizedBounds, NormalizedPoint, bounds_overlap, dataclass_wire, reject_if
from agent.domain.verified_text_transaction import VerifiedTextTransactionError, plan_from_input_states


UNIVERSAL_CONTROLLER_PROTOCOL_VERSION = "2026-09-03-universal-action-v22"

LOCAL_POINT_GROUNDING_SOURCE = "2026-09-03-stable-local-visual-surface-v3"
MAX_LOCAL_GROUNDING_BOX_GAP = 0.06

GESTURE_EDGE_MARGIN = 0.02
TARGETED_SWIPE_EDGE_MARGIN = 0.08
ELEMENT_SWIPE_START_INSET_RATIO = 0.06
MIN_DRAG_DISTANCE = 0.08
MAX_DRAG_DISTANCE = 0.90
DRAG_DURATION_SECONDS = 0.8
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
            not isinstance(value, tuple) or len(value) != length or any(
                isinstance(part, bool) or not isinstance(part, (int, float)) or not math.isfinite(float(part))
                for part in value
            ),
            UniversalActionError(f"{label}格式无效。"),
        )
        numbers = tuple(float(part) for part in value)
        reject_if(not all(0.0 <= part <= 1.0 for part in numbers), UniversalActionError(f"{label}超出归一化画面。"))
        reject_if(
            length == 4 and not (numbers[0] < numbers[2] and numbers[1] < numbers[3]),
            UniversalActionError(f"{label}超出归一化画面。"),
        )
        return numbers

    def validate_for(self, scene: UIScene, element: UIElement) -> None:
        reject_if(self.source != LOCAL_POINT_GROUNDING_SOURCE, UniversalActionError("本地落点证据来源无效。"))
        reject_if(self.scene_fingerprint != scene.fingerprint, UniversalActionError("本地落点证据不属于当前新鲜画面。"))
        reject_if(
            self.element_id != element.element_id or self.label != element.label,
            UniversalActionError("本地落点证据没有绑定当前唯一目标。"),
        )
        reject_if(
            element.element_id.startswith("local_audited_")
            or element.meaning == "application_text_input"
            or element.meaning.startswith(("input_", "ime_", "switch_keyboard_")),
            UniversalActionError("输入事务目标不允许使用普通视觉落点修正。"),
        )
        model_bounds = self._normalized_numbers(self.model_bounds, 4, "模型目标框")
        grounded_bounds = self._normalized_numbers(self.grounded_bounds, 4, "本地视觉框")
        proposed = self._normalized_numbers(self.proposed_point, 2, "模型提议落点")
        grounded = self._normalized_numbers(self.grounded_point, 2, "本地修正落点")
        reject_if(
            any(abs(actual - expected) > 1e-9 for actual, expected in zip(model_bounds, element.bounds))
            or any(abs(actual - expected) > 1e-9 for actual, expected in zip(proposed, element.center)),
            UniversalActionError("本地落点证据与当前目标几何不一致。"),
        )
        left, top, right, bottom = grounded_bounds
        reject_if(
            not (left <= grounded[0] <= right and top <= grounded[1] <= bottom),
            UniversalActionError("本地修正落点不在已识别视觉框内。"),
        )
        reject_if(
            isinstance(self.matched_frames, bool)
            or isinstance(self.inspected_frames, bool)
            or not isinstance(self.matched_frames, int)
            or not isinstance(self.inspected_frames, int)
            or not 2 <= self.matched_frames <= self.inspected_frames <= 4,
            UniversalActionError("本地落点证据缺少至少两帧稳定匹配。"),
        )
        model_left, model_top, model_right, model_bottom = model_bounds
        grounded_left, grounded_top, grounded_right, grounded_bottom = grounded_bounds
        horizontal_gap = max(0.0, grounded_left - model_right, model_left - grounded_right)
        vertical_gap = max(0.0, grounded_top - model_bottom, model_top - grounded_bottom)
        reject_if(
            math.hypot(horizontal_gap, vertical_gap) > MAX_LOCAL_GROUNDING_BOX_GAP,
            UniversalActionError("本地视觉框与模型目标框不属于同一邻近区域。"),
        )


@dataclass(frozen=True)
class ResolvedSemanticAction:
    """One executable action bound to one fresh scene."""

    node_id: str
    kind: str
    normalized_point: NormalizedPoint | None = None
    normalized_end_point: NormalizedPoint | None = None
    text: str | None = None
    input_fragment: str | None = None
    input_method: str | None = None
    input_pinyin: str | None = None
    text_transport: str | None = None
    input_field_id: str | None = None
    prior_input_value: str | None = None
    expected_input_value: str | None = None
    expected_input_state: dict[str, Any] = field(default_factory=dict)
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
    proposed_normalized_point: NormalizedPoint | None = None
    point_grounding: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        value = dataclass_wire(self)
        for key in ("proposed_normalized_point", "point_grounding"):
            if value[key] is None:
                value.pop(key)
        return value


class UniversalActionController:
    """Bind Qwen's one current-frame action to safe executable geometry."""

    def resolve_one(
        self,
        action: SemanticAction,
        scene: UIScene,
        *,
        confirmed: bool = False,
        local_point_grounding: LocalPointGrounding | None = None,
    ) -> ResolvedSemanticAction:
        del confirmed  # Product confirmation is decided before this geometry-only service.
        scene.validate()
        reject_if(not scene.stable, UniversalActionError("页面仍在变化，不能执行动作。"))
        reject_if(
            local_point_grounding is not None
            and action.action not in {"tap_semantic", "dismiss_overlay", "double_tap", "long_press"},
            UniversalActionError("本地视觉落点只能修正当前已选中的普通点按目标。"),
        )

        def resolved(kind: str | None = None, **values: Any) -> ResolvedSemanticAction:
            return ResolvedSemanticAction(
                node_id=action.node_id,
                kind=kind or action.action,
                before_fingerprint=scene.fingerprint,
                **values,
            )

        if action.action == "tap_semantic":
            element = self._resolve_target(action, scene)
            auxiliary = None
            if element.meaning in {
                "ime_exact_candidate",
                "input_exact_literal_key",
                "switch_keyboard_layout",
                "switch_keyboard_case",
                "switch_keyboard_input_mode",
            }:
                auxiliary = self._validate_input_auxiliary_tap(element, scene)
            if element.meaning == "input_next_field_key":
                self._validate_next_field_tap(element)
            point_action = self._point_action(
                action,
                element,
                scene.fingerprint,
                scene=scene,
                local_point_grounding=local_point_grounding,
            )
            if auxiliary is None:
                return point_action
            input_element, prior, expected, expected_state = auxiliary
            return replace(
                point_action,
                input_method=element.meaning,
                prior_input_value=prior,
                expected_input_value=expected,
                expected_input_state=expected_state,
                input_element_id=input_element.element_id,
            )

        if action.action == "press_enter":
            element = self._resolve_target(action, scene)
            reject_if(
                element.meaning != "input_exact_enter_key",
                UniversalActionError("press_enter 必须绑定本地审计的唯一换行键。"),
            )
            input_element, prior, expected, expected_state = self._validate_input_auxiliary_tap(element, scene)
            point_action = self._point_action(action, element, scene.fingerprint)
            return replace(
                point_action,
                input_method=element.meaning,
                prior_input_value=prior,
                expected_input_value=expected,
                expected_input_state=expected_state,
                input_element_id=input_element.element_id,
            )

        if action.action == "dismiss_overlay":
            element = self._resolve_target(action, scene)
            return self._point_action(
                action,
                element,
                scene.fingerprint,
                scene=scene,
                local_point_grounding=local_point_grounding,
            )

        if action.action == "input_verified_text":
            return self._resolve_verified_input(action, scene, resolved)

        if action.action == "clear_verified_text":
            return self._resolve_verified_clear(action, scene, resolved)

        if action.action == "double_tap":
            element = self._resolve_target(action, scene)
            self._validate_gesture_point(element.center, label="双击落点")
            return self._point_action(
                action,
                element,
                scene.fingerprint,
                scene=scene,
                local_point_grounding=local_point_grounding,
            )

        if action.action == "long_press":
            element = self._resolve_target(action, scene)
            self._validate_gesture_point(element.center, label="长按落点")
            duration_ms = action.params.get("duration_ms", 800)
            reject_if(
                isinstance(duration_ms, bool) or not isinstance(duration_ms, (int, float)),
                UniversalActionError("长按 duration_ms 格式无效。"),
            )
            reject_if(
                not 500 <= float(duration_ms) <= 2000,
                UniversalActionError("长按 duration_ms 必须在500～2000之间。"),
            )
            point_action = self._point_action(
                action,
                element,
                scene.fingerprint,
                scene=scene,
                local_point_grounding=local_point_grounding,
            )
            return replace(point_action, hold_seconds=float(duration_ms) / 1000.0)

        if action.action == "drag":
            source = self._resolve_target(action, scene, prefix="source_")
            destination = self._resolve_target(action, scene, prefix="destination_")
            reject_if(
                source.element_id == destination.element_id,
                UniversalActionError("拖动起点和终点不能是同一元素。"),
            )
            self._validate_executable_element(source)
            self._validate_executable_element(destination)
            self._validate_gesture_point(source.center, label="拖动起点")
            self._validate_gesture_point(destination.center, label="拖动终点")
            distance = math.dist(source.center, destination.center)
            reject_if(
                not MIN_DRAG_DISTANCE <= distance <= MAX_DRAG_DISTANCE,
                UniversalActionError(
                    f"拖动两端中心距离必须在{MIN_DRAG_DISTANCE:.2f}～{MAX_DRAG_DISTANCE:.2f}个归一化屏幕单位之间。"
                ),
            )
            return resolved(
                "drag",
                normalized_point=source.center,
                normalized_end_point=destination.center,
                target_element_id=source.element_id,
                destination_element_id=destination.element_id,
                hold_seconds=DRAG_DURATION_SECONDS,
                path_distance=distance,
            )

        if action.action == "reveal_system_navigation":
            reject_if(action.params, UniversalActionError("系统导航栏唤出动作不能携带参数。"))
            self._require_hidden_immersive_navigation(scene)
            return resolved("reveal_system_navigation")

        if action.action == "launch_app":
            allowed = {
                "target_surface_id",
                "target_app_id",
                "target_app_name",
                "launch_ref",
                "expected_app_id",
            }
            reject_if(set(action.params) != allowed, UniversalActionError("App 直启动作包含协议外字段。"))
            launch_ref = str(action.params.get("launch_ref") or "").strip()
            expected_app_id = str(action.params.get("expected_app_id") or "").strip()
            target_app_id = str(action.params.get("target_app_id") or "").strip()
            target_app_name = str(action.params.get("target_app_name") or "").strip()
            reject_if(
                not launch_ref or not expected_app_id or not target_app_id or not target_app_name,
                UniversalActionError("App 直启没有绑定受信任引用、目标包和 typed App 视觉身份。"),
            )
            reject_if(
                not str(action.params.get("target_surface_id") or "").strip(),
                UniversalActionError("App 直启缺少 target_surface_id。"),
            )
            return resolved(
                "launch_app",
                launch_ref=launch_ref,
                expected_package_id=expected_app_id,
                target_app_id=target_app_id,
                target_app_name=target_app_name,
            )

        if action.action == "scroll":
            direction = str(action.params.get("direction") or "").strip().lower()
            reject_if(
                direction not in {"up", "down", "left", "right"},
                UniversalActionError(f"不支持的滑动方向：{direction}"),
            )
            reject_if(scene.screen_id == "system_recent_tasks" and direction not in {"up", "down"},
                UniversalActionError("系统后台任务页只允许上下 scroll 寻找目标。"))
            element_id = str(action.params.get("element_id") or "").strip()
            if element_id:
                element = self._resolve_target(action, scene)
                self._validate_executable_element(element)
                start, end = self._targeted_scroll_path(element, direction)
                return resolved(
                    "scroll",
                    normalized_point=start,
                    normalized_end_point=end,
                    direction=direction,
                    hold_seconds=DRAG_DURATION_SECONDS,
                    path_distance=math.dist(start, end),
                    target_element_id=element.element_id,
                )
            return resolved("scroll", direction=direction)

        if action.action == "swipe_element":
            element = self._resolve_target(action, scene)
            self._validate_executable_element(element)
            start = self._gesture_param_point(action.params.get("start"), "元素滑动起点")
            end = self._gesture_param_point(action.params.get("end"), "元素滑动终点")
            self._validate_element_swipe_start(element, start)
            self._validate_gesture_point(end, label="元素滑动终点")
            distance = math.dist(start, end)
            reject_if(not MIN_DRAG_DISTANCE <= distance <= MAX_DRAG_DISTANCE,
                UniversalActionError(
                    f"元素滑动轨迹距离必须在{MIN_DRAG_DISTANCE:.2f}～{MAX_DRAG_DISTANCE:.2f}之间。"))
            direction = self._gesture_direction(start, end)
            reject_if(scene.screen_id == "system_recent_tasks",
                UniversalActionError("系统后台任务页不得再用 swipe_element 移除卡片；清理任务必须点击当前截图中的系统一键清理按钮。"))
            return resolved("swipe_element", normalized_point=start, normalized_end_point=end,
                direction=direction, hold_seconds=DRAG_DURATION_SECONDS, path_distance=distance,
                target_element_id=element.element_id)

        if action.action in {"back", "home", "open_recent_apps", "wait_for_change"}:
            return resolved()

        raise UniversalActionError(f"通用动作控制器尚不支持：{action.action}")

    def _resolve_verified_input(self, action: SemanticAction, scene: UIScene, resolved: Any) -> ResolvedSemanticAction:
        try:
            text = normalize_user_text(action.params.get("text"), field_name="输入文字")
        except ValueError as exc:
            raise UniversalActionError(str(exc)) from exc
        element = self._resolve_target(action, scene, required_role="input")
        self._validate_executable_element(element)
        text_transport = str(action.params.get("text_transport") or "mechanical_keyboard")
        reject_if(
            text_transport not in {"mechanical_keyboard", "adb_keyboard"},
            UniversalActionError("文字输入 transport 无效。"),
        )
        input_field_id = str(element.states.get("input_field_id") or "").strip()
        reject_if(
            element.states.get("focused") is not True,
            UniversalActionError("文字输入前必须有当前画面证明输入框已聚焦。"),
        )

        if text_transport == "adb_keyboard":
            prior = element.states.get("value")
            fragment = action.params.get("input_fragment")
            expected = action.params.get("expected_input_value")
            reject_if(
                input_field_id in {"", "unknown"} or action.params.get("input_field_id") != input_field_id,
                UniversalActionError("ADB Keyboard 输入没有绑定当前 typed input_field_id。"),
            )
            reject_if(
                not isinstance(prior, str)
                or not isinstance(fragment, str)
                or not fragment
                or not isinstance(expected, str)
                or expected != prior + fragment
                or expected != text
                or action.params.get("prior_input_value") != prior,
                UniversalActionError("ADB Keyboard 输入的 prior/fragment/expected 与当前画面不一致。"),
            )
            reject_if(
                element.states.get("ime_preedit_text"),
                UniversalActionError("ADB Keyboard 输入前仍有未完成的输入法组合。"),
            )
            self._require_unique_input(
                scene,
                element,
                require_focused=True,
                require_empty_preedit=True,
                error="ADB Keyboard 输入要求当前画面只有一个同 typed 字段。",
            )
            return resolved(
                "input_verified_text",
                text=text,
                input_fragment=fragment,
                input_method="unicode_commit",
                text_transport=text_transport,
                prior_input_value=prior,
                expected_input_value=expected,
                input_field_id=input_field_id,
                target_element_id=element.element_id,
            )

        reject_if(
            element.states.get("keyboard_layout") != "qwerty",
            UniversalActionError("精确文字输入要求当前画面确认 QWERTY 键盘。"),
        )
        try:
            input_step = plan_from_input_states(text, element.states)
        except (ValueError, VerifiedTextTransactionError) as exc:
            raise UniversalActionError(f"无法建立精确文字输入事务：{exc}") from exc
        reject_if(input_step is None, UniversalActionError("输入框已经逐字等于目标文字，不得重复输入。"))
        reject_if(
            input_step.kind == "literal_key",
            UniversalActionError("下一分段需要独立可见的数字、空格或符号键审计，不能按字母键盘猜测。"),
        )
        reject_if(
            element.states.get("keyboard_input_mode") != input_step.required_mode,
            UniversalActionError("当前键盘输入模式与下一确定性文字分段不一致。"),
        )
        reject_if(
            input_step.required_case_mode
            and element.states.get("keyboard_case_mode") != input_step.required_case_mode,
            UniversalActionError("当前键盘大小写状态与下一确定性英文分段不一致。"),
        )
        reject_if(
            element.states.get("ime_preedit_text"),
            UniversalActionError("当前仍有未完成的输入法组合，禁止继续键入。"),
        )
        self._require_unique_input(
            scene,
            element,
            required_states={
                "keyboard_layout": "qwerty",
                "keyboard_input_mode": input_step.required_mode,
                **(
                    {"keyboard_case_mode": input_step.required_case_mode}
                    if input_step.required_case_mode
                    else {}
                ),
            },
            require_empty_preedit=True,
            error="精确文字输入要求当前画面只有一个同 typed 目标输入框。",
        )
        return resolved(
            "input_verified_text",
            normalized_point=element.center,
            text=text,
            input_fragment=input_step.segment,
            input_method=input_step.kind,
            input_pinyin=input_step.pinyin or None,
            text_transport=text_transport,
            prior_input_value=input_step.current_text,
            expected_input_value=input_step.expected_value,
            input_field_id=input_field_id or None,
            target_element_id=element.element_id,
        )

    def _resolve_verified_clear(self, action: SemanticAction, scene: UIScene, resolved: Any) -> ResolvedSemanticAction:
        element = self._resolve_target(action, scene, required_role="input")
        self._validate_executable_element(element)
        observed_value = element.states.get("value")
        observed_preedit = element.states.get("ime_preedit_text", "")
        text_transport = str(action.params.get("text_transport") or "mechanical_keyboard")
        input_field_id = str(element.states.get("input_field_id") or "").strip()
        reject_if(
            text_transport not in {"mechanical_keyboard", "adb_keyboard"},
            UniversalActionError("清空文字 transport 无效。"),
        )
        reject_if(
            element.states.get("focused") is not True,
            UniversalActionError("清空文字前必须有当前画面证明输入框已聚焦。"),
        )
        reject_if(not isinstance(observed_value, str), UniversalActionError("清空文字要求当前画面提供精确 states.value。"))
        reject_if(not isinstance(observed_preedit, str), UniversalActionError("清空文字的输入法预编辑状态格式无效。"))
        reject_if(
            not observed_value and not observed_preedit,
            UniversalActionError("清空文字要求应用值或输入法预编辑至少一项非空。"),
        )
        extra_delete_units = element.states.get("clear_extra_delete_units", 0)
        reject_if(
            isinstance(extra_delete_units, bool)
            or not isinstance(extra_delete_units, int)
            or not 0 <= extra_delete_units <= 30,
            UniversalActionError("清空文字的额外视觉行退格单位无效。"),
        )
        delete_count = None
        if text_transport == "mechanical_keyboard":
            delete_count = (
                editable_character_count(observed_value)
                + editable_character_count(observed_preedit)
                + extra_delete_units
            )
            reject_if(
                not 1 <= delete_count <= 100,
                UniversalActionError("清空文字的已验证字符数必须在1～100之间。"),
            )
        self._require_unique_input(
            scene,
            element,
            require_focused=True,
            require_text=True,
            error="清空文字要求当前画面只有一个同 typed 非空目标输入框。",
        )
        if text_transport == "adb_keyboard":
            reject_if(
                input_field_id in {"", "unknown"}
                or action.params.get("input_field_id") != input_field_id
                or action.params.get("prior_input_value") != observed_value
                or action.params.get("expected_input_value") != "",
                UniversalActionError("ADB Keyboard 清空没有绑定当前 typed 文字事务。"),
            )
        return resolved(
            "clear_verified_text",
            normalized_point=None if text_transport == "adb_keyboard" else element.center,
            text="",
            delete_count=delete_count,
            text_transport=text_transport,
            input_field_id=input_field_id or None,
            prior_input_value=observed_value,
            expected_input_value="",
            target_element_id=element.element_id,
        )

    def verify_after_action(
        self,
        resolved: ResolvedSemanticAction,
        before: UIScene,
        after: UIScene,
    ) -> tuple[str, ...]:
        """Check only exact local receipts; Qwen owns all goal progress."""

        before.validate()
        after.validate()
        if resolved.kind != "wait_for_change":
            reject_if(not resolved.before_fingerprint, UniversalActionError("动作缺少执行前场景 fingerprint。"))
            reject_if(
                resolved.before_fingerprint != before.fingerprint,
                UniversalActionError("动作绑定的 fingerprint 已过期。"),
            )
        reject_if(not after.stable, UniversalActionError("动作后的页面仍在变化。"))

        if resolved.kind == "launch_app":
            observed_app = str(after.foreground_app_id or "").strip().casefold()
            package_match = bool(
                resolved.expected_package_id
                and observed_app == resolved.expected_package_id.strip().casefold()
            )
            reject_if(
                not package_match
                and not scene_matches_app_identity(
                    after,
                    resolved.target_app_id or "",
                    resolved.target_app_name or "",
                ),
                UniversalActionError(f"动作后画面不能证明目标 App 已在前台：{after.foreground_app_id}。"),
            )
            return (f"控制器确认目标 App 已在前台：{after.foreground_app_id}",)

        if resolved.kind == "reveal_system_navigation":
            self._require_hidden_immersive_navigation(before)
            reject_if(
                after.system_ui.navigation_bar_visible is not True,
                UniversalActionError("动作后缺少结构化导航栏可见证据。"),
            )
            return ("控制器确认系统导航栏已显示",)

        if (
            resolved.kind in {"input_verified_text", "press_enter", "clear_verified_text"}
            or resolved.input_element_id
        ):
            preedit_pending = self._verify_exact_input_value(resolved, before, after)
            if preedit_pending:
                return (CONTROLLER_INPUT_PREEDIT_PENDING,)
            return (f"控制器确认 typed 输入值：{resolved.expected_input_value!r}",)

        # Point gestures, swipes and system navigation do not prove task
        # semantics here.  A stable fresh screenshot is the only receipt; Qwen
        # decides the next action or finish from that screenshot.
        return ()

    def _require_unique_input(
        self,
        scene: UIScene,
        target: UIElement,
        *,
        required_states: Mapping[str, Any] | None = None,
        require_focused: bool = True,
        require_empty_preedit: bool = False,
        require_text: bool = False,
        error: str,
    ) -> None:
        required_states = required_states or {}
        typed_field_id = str(target.states.get("input_field_id") or "").strip()
        candidates = tuple(
            element
            for element in scene.elements
            if element.role == "input"
            and (
                element.element_id == target.element_id
                or (
                    typed_field_id not in {"", "unknown"}
                    and str(element.states.get("input_field_id") or "").strip() == typed_field_id
                )
            )
            and (not require_focused or element.states.get("focused") is True)
            and all(element.states.get(key) == value for key, value in required_states.items())
            and (not require_empty_preedit or not element.states.get("ime_preedit_text"))
            and (
                not require_text
                or (
                    isinstance(element.states.get("value"), str)
                    and isinstance(element.states.get("ime_preedit_text", ""), str)
                    and bool(element.states.get("value") or element.states.get("ime_preedit_text"))
                )
            )
        )
        reject_if(
            len(candidates) != 1 or candidates[0].element_id != target.element_id,
            UniversalActionError(error),
        )

    @staticmethod
    def _validate_next_field_tap(element: UIElement) -> None:
        states = element.states
        source_id = str(states.get("source_input_field_id") or "").strip()
        target_id = str(states.get("target_input_field_id") or "").strip()
        reject_if(
            not (
                element.role == "button"
                and states.get("input_next_field_key") is True
                and states.get("key_action") == "next"
                and source_id
                and target_id
                and source_id != target_id
            ),
            UniversalActionError("Next键没有绑定唯一 typed 字段切换。"),
        )

    def _validate_input_auxiliary_tap(
        self,
        element: UIElement,
        scene: UIScene,
    ) -> tuple[UIElement, str, str, dict[str, Any]]:
        states = element.states
        reject_if(
            element.role != "button",
            UniversalActionError("输入辅助键在当前画面不可执行。"),
        )
        input_id = str(states.get("input_element_id") or "").strip()
        input_element = self._required_element(scene, input_id, error="输入辅助键没有绑定唯一输入框")
        prior = states.get("prior_input_value")
        reject_if(
            input_element.role != "input"
            or input_element.states.get("focused") is not True
            or not isinstance(prior, str)
            or input_element.states.get("value") != prior,
            UniversalActionError("输入辅助键与当前精确输入前缀不一致。"),
        )

        if element.meaning == "ime_exact_candidate":
            expected = states.get("expected_input_value")
            reject_if(
                states.get("ime_candidate") is not True
                or not isinstance(expected, str)
                or expected != prior + element.label,
                UniversalActionError("输入法候选没有绑定唯一下一正文和精确结果。"),
            )
            return input_element, prior, expected, {}

        if element.meaning == "input_exact_literal_key":
            key_value = states.get("key_value")
            expected = states.get("expected_input_value")
            reject_if(
                states.get("input_literal_key") is not True
                or not isinstance(key_value, str)
                or len(key_value) != 1
                or expected != prior + key_value,
                UniversalActionError("逐键输入没有绑定唯一下一字符和精确结果。"),
            )
            return input_element, prior, expected, {}

        if element.meaning == "input_exact_enter_key":
            expected = states.get("expected_input_value")
            reject_if(
                states.get("input_enter_key") is not True
                or states.get("key_action") != "newline"
                or states.get("key_value") != "\n"
                or expected != prior + "\n",
                UniversalActionError("换行键没有绑定 multiline newline 与精确输入结果。"),
            )
            reject_if(
                input_element.states.get("input_multiline") is not True,
                UniversalActionError("当前输入框没有本地多行字段凭据。"),
            )
            return input_element, prior, expected, {}

        specs = {
            "switch_keyboard_layout": (
                "keyboard_layout_switch",
                "current_layout",
                "target_layout",
                "keyboard_layout",
                {"qwerty", "numeric", "symbol"},
                {},
                "键盘布局切换方向无效。",
            ),
            "switch_keyboard_case": (
                "keyboard_case_switch",
                "current_mode",
                "target_mode",
                "keyboard_case_mode",
                {"lower", "upper"},
                {"keyboard_layout": "qwerty", "keyboard_input_mode": "direct_latin"},
                "键盘大小写切换方向无效。",
            ),
            "switch_keyboard_input_mode": (
                "keyboard_input_mode_switch",
                "current_mode",
                "target_mode",
                "keyboard_input_mode",
                {"direct_latin", "chinese_pinyin"},
                {"keyboard_layout": "qwerty"},
                "键盘输入模式切换方向无效。",
            ),
        }
        reject_if(element.meaning not in specs, UniversalActionError("未知输入辅助键。"))
        flag, current_key, target_key, state_key, modes, fixed_states, message = specs[element.meaning]
        current = states.get(current_key)
        target = states.get(target_key)
        reject_if(
            states.get(flag) is not True
            or current not in modes
            or target not in modes
            or current == target
            or input_element.states.get(state_key) != current
            or any(input_element.states.get(key) != value for key, value in fixed_states.items()),
            UniversalActionError(message),
        )
        return input_element, prior, prior, {state_key: target}

    @staticmethod
    def _require_hidden_immersive_navigation(scene: UIScene) -> Any:
        system_ui = scene.system_ui
        reject_if(
            system_ui.immersive_or_fullscreen is not True
            or system_ui.navigation_bar_visible is not False,
            UniversalActionError("系统导航栏唤出动作要求当前画面明确处于沉浸态且导航栏隐藏。"),
        )
        return system_ui

    @staticmethod
    def _validate_gesture_point(point: NormalizedPoint, *, label: str) -> None:
        x, y = point
        reject_if(
            not (
                GESTURE_EDGE_MARGIN <= x <= 1.0 - GESTURE_EDGE_MARGIN
                and GESTURE_EDGE_MARGIN <= y <= 1.0 - GESTURE_EDGE_MARGIN
            ),
            UniversalActionError(
                f"{label}必须离画面边缘至少{GESTURE_EDGE_MARGIN:.2f}个归一化屏幕单位。"
            ),
        )

    @staticmethod
    def _validate_executable_element(element: UIElement) -> None:
        # Qwen already selected this exact current-frame element.  Optional
        # model-authored display-state hints are diagnostic facts, not a
        # second authority over an otherwise valid canonical action.
        element.validate()

    @classmethod
    def _targeted_scroll_path(cls, element: UIElement, direction: str) -> tuple[NormalizedPoint, NormalizedPoint]:
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
        distance = math.dist(start, end)
        reject_if(
            not MIN_DRAG_DISTANCE <= distance <= MAX_DRAG_DISTANCE,
            UniversalActionError(
                f"元素滑动轨迹距离必须在{MIN_DRAG_DISTANCE:.2f}～{MAX_DRAG_DISTANCE:.2f}之间。"
            ),
        )
        return start, end

    @staticmethod
    def _gesture_param_point(value: Any, label: str) -> NormalizedPoint:
        reject_if(not isinstance(value, (list, tuple)) or len(value) != 2
            or any(isinstance(part, bool) or not isinstance(part, (int, float))
            or not math.isfinite(float(part)) for part in value),
            UniversalActionError(f"{label}格式无效。"))
        point = (float(value[0]), float(value[1]))
        reject_if(not all(0.0 <= part <= 1.0 for part in point),
            UniversalActionError(f"{label}超出归一化画面。"))
        return point

    @classmethod
    def _validate_element_swipe_start(cls, element: UIElement, point: NormalizedPoint) -> None:
        cls._validate_gesture_point(point, label="元素滑动起点")
        left, top, right, bottom = element.bounds
        inset_x = (right - left) * ELEMENT_SWIPE_START_INSET_RATIO
        inset_y = (bottom - top) * ELEMENT_SWIPE_START_INSET_RATIO
        reject_if(not (left + inset_x <= point[0] <= right - inset_x
            and top + inset_y <= point[1] <= bottom - inset_y),
            UniversalActionError("元素滑动起点必须位于当前目标元素内部安全区域。"))

    @staticmethod
    def _gesture_direction(start: NormalizedPoint, end: NormalizedPoint) -> str:
        delta_x = end[0] - start[0]
        delta_y = end[1] - start[1]
        reject_if(abs(delta_x) == abs(delta_y),
            UniversalActionError("元素滑动轨迹方向不唯一。"))
        if abs(delta_x) > abs(delta_y):
            return "right" if delta_x > 0 else "left"
        return "down" if delta_y > 0 else "up"

    @staticmethod
    def _regions_stably_overlap(before_bounds: NormalizedBounds, after_bounds: NormalizedBounds) -> bool:
        return bounds_overlap(before_bounds, after_bounds)["intersection_over_smaller"] >= 0.60

    @staticmethod
    def _required_element(scene: UIScene, element_id: str, *, error: str) -> UIElement:
        try:
            return scene.get_element(element_id)
        except UISceneError as exc:
            raise UniversalActionError(f"{error}：{exc}") from exc

    def _verify_exact_input_value(
        self,
        resolved: ResolvedSemanticAction,
        before: UIScene,
        after: UIScene,
    ) -> bool:
        expected = resolved.expected_input_value
        target_id = str(resolved.input_element_id or resolved.target_element_id or "").strip()
        reject_if(expected is None or not target_id, UniversalActionError("输入动作缺少精确文字或目标输入框身份。"))
        reject_if(
            resolved.kind == "input_verified_text"
            and (not expected or not resolved.input_fragment or not resolved.input_method),
            UniversalActionError("输入动作缺少精确文字或目标输入框身份。"),
        )
        reject_if(
            resolved.kind == "press_enter"
            and (resolved.prior_input_value is None or expected != resolved.prior_input_value + "\n"),
            UniversalActionError("换行动作缺少精确前缀或 newline 后置值。"),
        )
        if resolved.kind == "clear_verified_text":
            reject_if(
                expected != ""
                or (
                    (resolved.text_transport or "mechanical_keyboard") == "mechanical_keyboard"
                    and resolved.delete_count is None
                ),
                UniversalActionError("清空动作缺少空值或精确退格次数。"),
            )

        before_input = self._required_element(before, target_id, error="输入前目标证据无效")
        reject_if(before_input.role != "input", UniversalActionError("输入前目标不是 input 元素。"))
        typed_field_id = str(before_input.states.get("input_field_id") or "").strip()
        if typed_field_id not in {"", "unknown"}:
            candidates = tuple(
                element
                for element in after.elements
                if element.role == "input"
                and str(element.states.get("input_field_id") or "").strip() == typed_field_id
            )
        else:
            candidates = tuple(
                element
                for element in after.elements
                if element.role == "input"
                and (element.element_id == before_input.element_id
                or self._regions_stably_overlap(before_input.bounds, element.bounds)
                )
            )
        reject_if(len(candidates) != 1, UniversalActionError("动作后无法唯一绑定原目标输入框。"))
        after_input = candidates[0]
        states = after_input.states
        reject_if(
            "value" not in states or not isinstance(states["value"], str),
            UniversalActionError("动作后缺少输入框 states.value 精确文字证据。"),
        )

        actual = states["value"]
        preedit_pending = False
        if resolved.input_method == "chinese_pinyin":
            reject_if(
                actual != resolved.prior_input_value,
                UniversalActionError("拼音键入后应用输入值在候选确认前已意外变化。"),
            )
            reject_if(
                states.get("ime_preedit_text") != resolved.input_pinyin,
                UniversalActionError("动作后缺少逐字一致的拼音组合证据。"),
            )
            reject_if(
                states.get("ime_exact_candidate_text") != resolved.input_fragment,
                UniversalActionError("动作后缺少唯一逐字一致的中文候选。"),
            )
            preedit_pending = True
        elif actual != expected:
            raise UniversalActionError(f"动作后输入框文字不匹配：实际 {actual!r}，预期 {expected!r}。")

        for key, value in resolved.expected_input_state.items():
            reject_if(
                states.get(key) != value,
                UniversalActionError(f"动作后输入框 {key} 不匹配：实际 {states.get(key)!r}，预期 {value!r}。"),
            )
        if resolved.text_transport == "adb_keyboard":
            reject_if(
                states.get("ime_preedit_text") not in (None, ""),
                UniversalActionError("ADB Keyboard 动作后仍残留输入法预编辑文字。"),
            )
        if resolved.kind == "clear_verified_text" and before_input.states.get("ime_preedit_text"):
            reject_if(
                states.get("ime_preedit_text") not in (None, ""),
                UniversalActionError("清空动作后输入法预编辑文字仍未清除。"),
            )
            reject_if(
                states.get("focused") is not True,
                UniversalActionError("清空动作后原 typed 输入框不再聚焦。"),
            )
        return preedit_pending

    def _resolve_target(
        self,
        action: SemanticAction,
        scene: UIScene,
        *,
        prefix: str = "",
        required_role: str | None = None,
    ) -> UIElement:
        element_id = str(action.params.get(f"{prefix}element_id") or "").strip()
        reject_if(
            not element_id,
            UniversalActionError(f"{action.action} 缺少 Qwen 同帧 {prefix}element_id。"),
        )
        try:
            element = scene.get_element(element_id)
            reject_if(
                required_role is not None and element.role != required_role,
                UISceneError(f"元素 {element_id} 的角色必须为 {required_role}。"),
            )
            return element
        except UISceneError as exc:
            raise UniversalActionError(str(exc)) from exc

    def _point_action(
        self,
        action: SemanticAction,
        element: UIElement,
        before_fingerprint: str,
        *,
        scene: UIScene | None = None,
        local_point_grounding: LocalPointGrounding | None = None,
    ) -> ResolvedSemanticAction:
        self._validate_executable_element(element)
        normalized_point = element.center
        proposed_point: NormalizedPoint | None = None
        grounding_payload: dict[str, Any] | None = None
        if local_point_grounding is not None:
            reject_if(scene is None, UniversalActionError("本地落点证据缺少当前场景绑定。"))
            local_point_grounding.validate_for(scene, element)
            proposed_point = element.center
            normalized_point = local_point_grounding.grounded_point
            grounding_payload = local_point_grounding.to_dict()
            if action.action in {"double_tap", "long_press"}:
                self._validate_gesture_point(normalized_point, label="本地修正手势落点")
        return ResolvedSemanticAction(
            node_id=action.node_id,
            kind=action.action,
            normalized_point=normalized_point,
            target_element_id=element.element_id,
            before_fingerprint=before_fingerprint,
            proposed_normalized_point=proposed_point,
            point_grounding=grounding_payload,
        )
