"""Pure domain service that validates and resolves one current-frame action.

The controller is deliberately not a second goal or visual authority.  Qwen
selects one action from the current screenshot.  This service only proves that
the selected action is executable on that same scene, resolves its geometry,
and checks exact text receipts. Post-action App identity and navigation progress
belong to Qwen's same-response action/finish, not a second local visual verdict.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
from typing import Any

from agent.domain.semantic_action import SemanticAction
from agent.domain.text_input_utils import normalize_user_text
from agent.domain.ui_scene import UIElement, UIScene, UISceneError
from agent.domain.validation import DataclassWire, NormalizedPoint, bounds_overlap, dataclass_wire, reject_if


UNIVERSAL_CONTROLLER_PROTOCOL_VERSION = "2026-09-06-universal-adb-text-v28"

DRAG_DURATION_SECONDS = 0.8


class UniversalActionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResolvedSemanticAction:
    """One executable action bound to one fresh scene."""

    node_id: str
    kind: str
    normalized_point: NormalizedPoint | None = None
    normalized_end_point: NormalizedPoint | None = None
    text: str | None = None
    input_fragment: str | None = None
    text_transport: str | None = None
    input_field_id: str | None = None
    prior_input_value: str | None = None
    expected_input_value: str | None = None
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

    def to_dict(self) -> dict[str, Any]:
        return dataclass_wire(self)


class UniversalActionController:
    """Bind Qwen's one current-frame action to safe executable geometry."""

    def resolve_one(
        self,
        action: SemanticAction,
        scene: UIScene,
        *,
        confirmed: bool = False,
    ) -> ResolvedSemanticAction:
        del confirmed  # Product confirmation is decided before this geometry-only service.
        scene.validate()
        def resolved(kind: str | None = None, **values: Any) -> ResolvedSemanticAction:
            return ResolvedSemanticAction(
                node_id=action.node_id,
                kind=kind or action.action,
                before_fingerprint=scene.fingerprint,
                **values,
            )

        if action.action == "tap_semantic":
            element = self._resolve_direct_point_target(action, scene)
            return self._point_action(action, element, scene.fingerprint)

        if action.action == "press_enter":
            element = self._resolve_target(action, scene, required_role="input")
            reject_if(element.states.get("input_multiline") is not True
                or action.params.get("input_fragment") != "\n",
                UniversalActionError("换行必须绑定当前多行字段并只追加一个 newline。"))
            return replace(self._resolve_verified_input(action, scene, resolved), kind="press_enter")

        if action.action == "dismiss_overlay":
            element = self._resolve_direct_point_target(action, scene)
            return self._point_action(
                action,
                element,
                scene.fingerprint,
            )

        if action.action == "input_verified_text":
            return self._resolve_verified_input(action, scene, resolved)

        if action.action == "clear_verified_text":
            return self._resolve_verified_clear(action, scene, resolved)

        if action.action == "double_tap":
            element = self._resolve_direct_point_target(action, scene)
            point_action = self._point_action(
                action,
                element,
                scene.fingerprint,
            )
            self._validate_gesture_point(point_action.normalized_point, label="双击落点")
            return point_action

        if action.action == "long_press":
            element = self._resolve_direct_point_target(action, scene)
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
            )
            self._validate_gesture_point(point_action.normalized_point, label="长按落点")
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
                not math.isfinite(distance) or distance <= 0,
                UniversalActionError("拖动轨迹必须是有限非零距离。"),
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
            reject_if(not math.isfinite(distance) or distance <= 0,
                UniversalActionError("元素滑动轨迹必须是有限非零距离。"))
            direction = self._gesture_direction(start, end)
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
        text_transport = action.params.get("text_transport")
        reject_if(text_transport != "adb_keyboard", UniversalActionError("文字输入需要 ADB Keyboard。"))
        input_field_id = str(element.states.get("input_field_id") or "").strip()
        reject_if(
            element.states.get("focused") is not True,
            UniversalActionError("文字输入前必须有当前画面证明输入框已聚焦。"),
        )

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
            text_transport=text_transport,
            prior_input_value=prior,
            expected_input_value=expected,
            input_field_id=input_field_id,
            target_element_id=element.element_id,
        )


    def _resolve_verified_clear(self, action: SemanticAction, scene: UIScene, resolved: Any) -> ResolvedSemanticAction:
        element = self._resolve_target(action, scene, required_role="input")
        self._validate_executable_element(element)
        observed_value = element.states.get("value")
        observed_preedit = element.states.get("ime_preedit_text", "")
        text_transport = action.params.get("text_transport")
        input_field_id = str(element.states.get("input_field_id") or "").strip()
        reject_if(
            text_transport != "adb_keyboard",
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
            text="",
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

        if (
            resolved.kind in {"input_verified_text", "press_enter", "clear_verified_text"}
            or resolved.input_element_id
        ):
            self._verify_exact_input_value(resolved, before, after)
            return (f"控制器确认 typed 输入值：{resolved.expected_input_value!r}",)

        # Launches, point gestures, swipes and system navigation do not prove task
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
        UniversalActionController._gesture_param_point(point, label)

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
            end = (center_x, 0.0)
        elif direction == "down":
            start = (center_x, top + height * 0.25)
            end = (center_x, 1.0)
        elif direction == "left":
            start = (left + width * 0.75, center_y)
            end = (0.0, center_y)
        elif direction == "right":
            start = (left + width * 0.25, center_y)
            end = (1.0, center_y)
        else:
            raise UniversalActionError(f"不支持的元素滑动方向：{direction}")
        cls._validate_gesture_point(start, label="元素滑动起点")
        cls._validate_gesture_point(end, label="元素滑动终点")
        distance = math.dist(start, end)
        reject_if(
            not math.isfinite(distance) or distance <= 0,
            UniversalActionError("元素滑动轨迹必须是有限非零距离。"),
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
        reject_if(not (left <= point[0] <= right and top <= point[1] <= bottom),
            UniversalActionError("元素滑动起点必须位于当前目标元素内。"))

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
            and (not expected or not resolved.input_fragment),
            UniversalActionError("输入动作缺少精确文字或目标输入框身份。"),
        )
        reject_if(
            resolved.kind == "press_enter"
            and (resolved.prior_input_value is None or expected != resolved.prior_input_value + "\n"),
            UniversalActionError("换行动作缺少精确前缀或 newline 后置值。"),
        )
        if resolved.kind == "clear_verified_text":
            reject_if(
                expected != "",
                UniversalActionError("清空动作缺少空值。"),
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
        if actual != expected:
            raise UniversalActionError(f"动作后输入框文字不匹配：实际 {actual!r}，预期 {expected!r}。")

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
        return False

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

    def _resolve_direct_point_target(self, action: SemanticAction, scene: UIScene) -> UIElement | None:
        """Resolve only locally projected input controls; ordinary point targets have no bounds."""

        element_id = str(action.params.get("element_id") or "").strip()
        reject_if(not element_id, UniversalActionError(f"{action.action} 缺少 Qwen 同帧严格目标身份。"))
        try:
            element = scene.get_element(element_id)
        except UISceneError:
            return None
        declared_role = str(action.params.get("role") or "").strip()
        reject_if(bool(declared_role) and declared_role != element.role,
            UniversalActionError("Qwen点按目标与同帧本地投影元素角色不一致。"))
        return element

    def _point_action(
        self,
        action: SemanticAction,
        element: UIElement | None,
        before_fingerprint: str,
    ) -> ResolvedSemanticAction:
        if element is not None:
            self._validate_executable_element(element)
            target_element_id = element.element_id
        else:
            target_element_id = str(action.params.get("element_id") or "").strip()
            target = str(action.params.get("target") or "").strip()
            role = str(action.params.get("role") or "").strip()
            reject_if(not target_element_id or not target or not role,
                UniversalActionError("Qwen点按动作缺少同帧唯一目标身份。"))
        point = self._gesture_param_point(action.params.get("tap_point"), "Qwen明确点击点")
        return ResolvedSemanticAction(
            node_id=action.node_id,
            kind=action.action,
            normalized_point=point,
            target_element_id=target_element_id,
            before_fingerprint=before_fingerprint,
        )
