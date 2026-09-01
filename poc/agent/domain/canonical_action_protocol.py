"""Bind one Qwen action from the current screenshot to trusted local data.

Qwen owns the choice. This module never builds or ranks another semantic
candidate catalog after the model has selected an action.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .canonical_action_kinds import CANONICAL_ACTION_KINDS
from .semantic_action import SemanticAction
from .text_transport import TextTransportProfile
from .ui_scene import UIElement
from .validation import dataclass_wire, reject_if

CANONICAL_ACTION_PROTOCOL = "2026-08-20-canonical-action-v1"
MODEL_STEP_DECISION_FIELDS = frozenset({"status", "action", "element_id", "source_element_id",
    "destination_element_id", "direction", "evidence_refs", "confidence", "reason"})
MODEL_STEP_SINGLE_ELEMENT_ACTIONS = frozenset({"tap_semantic", "dismiss_overlay", "input_verified_text",
    "press_enter", "clear_verified_text", "double_tap", "long_press"})
_ACTION_INJECTION_FIELDS = frozenset({"actions", "plan", "plans", "step", "steps", "tap", "click", "swipe",
    "command", "shell", "coordinates", "coordinate", "x", "y", "next_action", "execution_plan"})


class CanonicalActionProtocolError(ValueError):
    pass


def _contains_action_injection(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(str(key).strip().casefold() in _ACTION_INJECTION_FIELDS
            or _contains_action_injection(part) for key, part in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_action_injection(part) for part in value)
    return False


def normalize_model_step_decision(value: Any) -> dict[str, Any]:
    """Normalize the one action-or-finish union shared by scene and action binding."""

    reject_if(not isinstance(value, Mapping), CanonicalActionProtocolError("同响应decision必须是对象。"))
    extras = {key: part for key, part in value.items() if key not in MODEL_STEP_DECISION_FIELDS}
    reject_if(_contains_action_injection(extras),
        CanonicalActionProtocolError("同响应decision包含多动作、计划或裸坐标字段。"))
    status = value.get("status")
    reject_if(status not in {"action", "finish"},
        CanonicalActionProtocolError("同响应decision只允许action或finish。"))
    confidence = value.get("confidence", 1.0)
    if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
        or not 0.0 <= float(confidence) <= 1.0):
        confidence = 1.0
    reason = value.get("reason", "")
    if not isinstance(reason, str):
        reason = ""
    raw_refs = value.get("evidence_refs", [])
    refs = list(dict.fromkeys(item.strip() for item in raw_refs
        if isinstance(item, str) and item.strip()))[:8] if isinstance(raw_refs, list) else []
    action = value.get("action")
    element_id = value.get("element_id")
    source_id = value.get("source_element_id")
    destination_id = value.get("destination_element_id")
    direction = value.get("direction")
    for name, part in {"action": action, "element_id": element_id, "source_element_id": source_id,
        "destination_element_id": destination_id, "direction": direction}.items():
        reject_if(part is not None and (not isinstance(part, str) or not part.strip()),
            CanonicalActionProtocolError(f"同响应decision.{name}必须是非空字符串或null。"))
    if status == "finish":
        reject_if(any(part is not None for part in (action, element_id, source_id, destination_id, direction))
            or not refs, CanonicalActionProtocolError("finish必须只引用同一scene完成证据。"))
    else:
        # evidence_refs is optional diagnostic metadata for an action. It never
        # becomes completion evidence or an executability gate.
        reject_if(action not in CANONICAL_ACTION_KINDS,
            CanonicalActionProtocolError("action必须是canonical动作。"))
        if action in MODEL_STEP_SINGLE_ELEMENT_ACTIONS:
            reject_if(element_id is None or any(part is not None for part in (source_id, destination_id, direction)),
                CanonicalActionProtocolError("元素动作必须且只能引用一个element_id。"))
        elif action == "drag":
            reject_if(element_id is not None or direction is not None or source_id is None or destination_id is None
                or source_id == destination_id,
                CanonicalActionProtocolError("drag必须且只能引用不同起点和终点。"))
        elif action == "swipe":
            reject_if(source_id is not None or destination_id is not None
                or direction not in {"up", "down", "left", "right"},
                CanonicalActionProtocolError("swipe必须声明唯一方向。"))
        else:
            reject_if(any(part is not None for part in (element_id, source_id, destination_id, direction)),
                CanonicalActionProtocolError("系统动作不得携带元素或方向字段。"))
    normalized_reason = reason.strip()[:500] or ("当前截图同帧证据证明目标完成" if status == "finish"
        else "当前截图选择一个推进目标的动作")
    return {"status": status, "action": action, "element_id": element_id,
        "source_element_id": source_id, "destination_element_id": destination_id, "direction": direction,
        "evidence_refs": list(refs), "confidence": float(confidence), "reason": normalized_reason}


@dataclass(frozen=True)
class GenericStepProposal:
    status: str
    action: SemanticAction | None = None
    reason: str = ""

    def validate(self, scene: Any) -> None:
        reject_if(self.status not in {"action", "finish"},
            CanonicalActionProtocolError("Qwen 单步只允许 action 或 finish。"))
        reject_if((self.status == "action") != (self.action is not None),
            CanonicalActionProtocolError("action/finish 与动作载荷不一致。"))
        if self.action is None:
            return
        reject_if(self.action.action not in CANONICAL_ACTION_KINDS,
            CanonicalActionProtocolError(f"未知 canonical 动作：{self.action.action}"))
        for key in ("element_id", "source_element_id", "destination_element_id"):
            element_id = str(self.action.params.get(key) or "").strip()
            if element_id:
                scene.get_element(element_id)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_wire(self)


def bind_same_response_action(payload: Mapping[str, Any], *, context: Any, observation: Any,
    available_action_kinds: Iterable[str], launch_target: Mapping[str, str] | None=None,
    text_transport_profile: TextTransportProfile | None=None) -> SemanticAction:
    """Bind Qwen's one current-frame choice without choosing another action."""

    scene = observation.scene
    scene.validate()
    kind = str(payload.get("action") or "").strip()
    available = frozenset(str(item).strip() for item in available_action_kinds)
    reject_if(kind not in CANONICAL_ACTION_KINDS or kind not in available,
        CanonicalActionProtocolError(f"当前设备不支持 Qwen 选择的 canonical 动作：{kind or 'missing'}。"))

    params: dict[str, Any] = {}
    point_actions = {"tap_semantic", "dismiss_overlay", "input_verified_text", "press_enter",
        "clear_verified_text", "double_tap", "long_press"}
    if kind in point_actions:
        element = _selected_element(observation, payload.get("element_id"), kind)
        _validate_typed_field_binding(kind, context=context, scene=scene, element=element)
        params.update(_element_params(element))
        if kind in {"input_verified_text", "clear_verified_text"}:
            params.update(_text_action_params(kind, context=context, element=element,
                profile=text_transport_profile))
        if kind == "long_press":
            params["duration_ms"] = 800
    elif kind == "drag":
        source = _selected_element(observation, payload.get("source_element_id"), kind)
        destination = _selected_element(observation, payload.get("destination_element_id"), kind)
        reject_if(source.element_id == destination.element_id,
            CanonicalActionProtocolError("drag 起点和终点不能相同。"))
        params.update(_element_params(source, prefix="source_"))
        params.update(_element_params(destination, prefix="destination_"))
    elif kind == "swipe":
        direction = str(payload.get("direction") or "").strip().lower()
        reject_if(direction not in {"up", "down", "left", "right"},
            CanonicalActionProtocolError("swipe 必须声明一个合法方向。"))
        params["direction"] = direction
        element_id = str(payload.get("element_id") or "").strip()
        if element_id:
            params.update(_element_params(_selected_element(observation, element_id, kind)))
    elif kind == "launch_app":
        params.update(_launch_params(launch_target))
    elif kind not in {"back", "home", "open_recent_apps", "reveal_system_navigation", "wait_for_change"}:
        raise CanonicalActionProtocolError(f"未实现的 canonical 动作：{kind}")

    return SemanticAction(node_id=f"qwen_visual_revision_{context.revision}", action=kind, params=params)


def _selected_element(observation: Any, value: Any, kind: str) -> UIElement:
    element_id = str(value or "").strip()
    reject_if(not element_id, CanonicalActionProtocolError(f"{kind} 缺少当前 scene element_id。"))
    try:
        element = observation.get_candidate(element_id)
    except Exception as exc:
        raise CanonicalActionProtocolError(f"Qwen 选择的当前元素不存在或不唯一：{element_id}") from exc
    element.validate()
    return element


def _element_params(element: UIElement, *, prefix: str="") -> dict[str, Any]:
    return {f"{prefix}element_id": element.element_id, f"{prefix}target": element.meaning,
        f"{prefix}role": element.role, f"{prefix}label": element.label,
        f"{prefix}states": dict(element.states)}


def _current_input_binding(context: Any) -> tuple[str, str]:
    subgoal = context.current_subgoal if isinstance(context.current_subgoal, Mapping) else {}
    field_id = str(subgoal.get("input_field_id") or "").strip()
    operation = str(subgoal.get("input_operation") or "").strip()
    reject_if(bool(field_id) != bool(operation),
        CanonicalActionProtocolError("当前子目标 typed 字段绑定不完整。"))
    return field_id, operation


def _validate_typed_field_binding(kind: str, *, context: Any, scene: Any, element: UIElement) -> None:
    field_id, operation = _current_input_binding(context)
    related = (element.role == "input" or bool(str(element.states.get("input_element_id") or "").strip())
        or kind in {"input_verified_text", "clear_verified_text", "press_enter"})
    if not related:
        return
    reject_if(not field_id, CanonicalActionProtocolError("输入动作缺少当前子目标 typed input_field_id。"))
    input_element = element
    if element.role != "input":
        input_id = str(element.states.get("input_element_id") or "").strip()
        reject_if(not input_id, CanonicalActionProtocolError("输入辅助动作没有绑定当前输入框。"))
        try:
            input_element = scene.get_element(input_id)
        except Exception as exc:
            raise CanonicalActionProtocolError("输入辅助动作绑定的输入框不在当前 scene。") from exc
    actual = str(input_element.states.get("input_field_id") or "").strip()
    reject_if(input_element.role != "input" or actual != field_id,
        CanonicalActionProtocolError("Qwen 所选输入元素与当前 typed input_field_id 不一致。"))
    if operation == "focus":
        reject_if(kind != "tap_semantic" or element.role != "input",
            CanonicalActionProtocolError("focus 子目标只能点击其 typed 输入框。"))
    elif operation == "clear_verified_text":
        reject_if(kind not in {"tap_semantic", "clear_verified_text"},
            CanonicalActionProtocolError("clear 子目标只能聚焦或清空其 typed 输入框。"))
    elif operation == "press_enter":
        reject_if(kind not in {"tap_semantic", "press_enter"},
            CanonicalActionProtocolError("newline 子目标只能聚焦或按当前输入框换行键。"))
    else:
        reject_if(operation != "input_verified_text" or kind not in {
            "tap_semantic", "input_verified_text", "clear_verified_text"},
            CanonicalActionProtocolError("当前 typed 输入操作与 Qwen 动作不一致。"))


def _typed_input_text(context: Any, element: UIElement) -> str:
    field_id, operation = _current_input_binding(context)
    reject_if(operation != "input_verified_text"
        or str(element.states.get("input_field_id") or "").strip() != field_id,
        CanonicalActionProtocolError("当前 Qwen 输入元素没有绑定本子目标 typed 字段。"))
    raw = context.requested_input_text
    reject_if(not isinstance(raw, str),
        CanonicalActionProtocolError("当前子目标没有唯一 typed 输入值。"))
    return raw


def _text_action_params(kind: str, *, context: Any, element: UIElement,
    profile: TextTransportProfile | None) -> dict[str, Any]:
    reject_if(element.role != "input" or element.states.get("focused") is not True,
        CanonicalActionProtocolError("文字动作必须引用当前已聚焦输入框。"))
    prior = element.states.get("value")
    reject_if(not isinstance(prior, str), CanonicalActionProtocolError("文字动作缺少当前输入值。"))
    field_id = str(element.states.get("input_field_id") or "").strip()
    expected_field_id, operation = _current_input_binding(context)
    reject_if(field_id != expected_field_id or (kind == "clear_verified_text"
        and operation not in {"clear_verified_text", "input_verified_text"}),
        CanonicalActionProtocolError("文字动作与当前 typed 字段或操作不一致。"))
    enabled_profile = None
    if profile is not None:
        profile.validate()
        reject_if(profile.device_id != context.device_id,
            CanonicalActionProtocolError("Companion IME profile 与当前设备不一致。"))
        if profile.enabled:
            enabled_profile = profile

    if kind == "clear_verified_text":
        reject_if(not prior and not str(element.states.get("ime_preedit_text") or ""),
            CanonicalActionProtocolError("当前输入框已经为空，不得重复清空。"))
        if enabled_profile is not None:
            reject_if("clear_text" not in enabled_profile.capabilities or field_id in {"", "unknown"},
                CanonicalActionProtocolError("当前 Companion IME 不能唯一清空该 typed 字段。"))
            return {"text_transport": "companion_ime", "input_field_id": field_id,
                "prior_input_value": prior, "expected_input_value": ""}
        return {"text_transport": "mechanical_keyboard", "input_field_id": field_id,
            "prior_input_value": prior, "expected_input_value": ""}

    target = _typed_input_text(context, element)
    reject_if(not target or "\r" in target, CanonicalActionProtocolError("typed 输入值为空或包含非法回车。"))
    if enabled_profile is not None:
        reject_if("append_text" not in enabled_profile.capabilities or field_id in {"", "unknown"}
            or not target.startswith(prior) or target == prior,
            CanonicalActionProtocolError("当前 Companion IME 不能建立唯一 prior/fragment/expected 事务。"))
        fragment = target[len(prior):]
        return {"text": target, "text_transport": "companion_ime", "input_field_id": field_id,
            "prior_input_value": prior, "input_fragment": fragment, "expected_input_value": target}
    return {"text": target, "text_transport": "mechanical_keyboard", "input_field_id": field_id,
        "prior_input_value": prior, "expected_input_value": target}


def _launch_params(launch_target: Mapping[str, str] | None) -> dict[str, Any]:
    required = {"launch_ref", "expected_app_id", "target_app_id", "target_app_name"}
    reject_if(not isinstance(launch_target, Mapping) or set(launch_target) != required
        or any(not isinstance(launch_target.get(key), str) or not str(launch_target[key]).strip()
        for key in required), CanonicalActionProtocolError("launch_app 缺少唯一可信包名映射。"))
    return {"target_surface_id": "target_app", "target_app_id": str(launch_target["target_app_id"]),
        "target_app_name": str(launch_target["target_app_name"]), "launch_ref": str(launch_target["launch_ref"]),
        "expected_app_id": str(launch_target["expected_app_id"])}


__all__ = ["CANONICAL_ACTION_PROTOCOL", "CanonicalActionProtocolError", "GenericStepProposal",
    "bind_same_response_action"]
