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

CANONICAL_ACTION_PROTOCOL = "2026-09-06-canonical-whole-task-v10"
MODEL_STEP_DECISION_FIELDS = frozenset({"status", "action", "element_id", "source_element_id",
    "destination_element_id", "direction", "target", "tap_point", "start", "end", "confidence",
    "reason", "text", "app", "previous_action_outcome", "state_action_consistent", "postcondition"})
MODEL_STEP_SINGLE_ELEMENT_ACTIONS = frozenset({"tap_semantic", "dismiss_overlay", "input_verified_text",
    "press_enter", "clear_verified_text", "double_tap", "long_press"})
MODEL_STEP_DIRECT_POINT_ACTIONS = frozenset({"tap_semantic", "dismiss_overlay",
    "double_tap", "long_press"})
_ACTION_INJECTION_FIELDS = frozenset({"actions", "plan", "plans", "step", "steps", "tap", "click", "swipe",
    "command", "shell", "coordinates", "coordinate", "x", "y", "next_action", "execution_plan"})
_DIRECT_TARGET_FIELDS = frozenset({"element_id", "role", "meaning", "label", "evidence"})



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
    action = value.get("action")
    text = value.get("text")
    app = value.get("app")
    outcome = value.get("previous_action_outcome")
    state_action_consistent = value.get("state_action_consistent")
    reject_if(state_action_consistent is False,
        CanonicalActionProtocolError("Qwen明确报告当前状态与所选动作矛盾。"))
    reject_if(state_action_consistent is not None and not isinstance(state_action_consistent, bool),
        CanonicalActionProtocolError("state_action_consistent必须是boolean或null。"))
    postcondition = value.get("postcondition")
    if outcome is not None and not isinstance(postcondition, Mapping):
        # Navigation/observation actions have no external effect state to
        # report.  Treat an omitted postcondition as not_applicable instead of
        # rejecting an already executed Home/back/recent-apps step.  Effectful
        # actions still require an explicit postcondition.
        if action in {"home", "back", "open_recent_apps", "reveal_system_navigation",
            "scroll", "wait_for_change", "launch_app"}:
            postcondition = {"status": "not_applicable", "fact": ""}
        else:
            raise CanonicalActionProtocolError("动作后必须提供postcondition。")
    if postcondition is not None:
        reject_if(not isinstance(postcondition, Mapping), CanonicalActionProtocolError("postcondition必须是对象或null。"))
        post_status = postcondition.get("status")
        reject_if(post_status not in {"confirmed", "not_confirmed", "unknown", "not_applicable"},
            CanonicalActionProtocolError("postcondition.status无效。"))
        reject_if(not isinstance(postcondition.get("fact", ""), str),
            CanonicalActionProtocolError("postcondition.fact必须是字符串。"))
    reject_if(outcome not in {None, "matched", "unmatched", "uncertain"},
        CanonicalActionProtocolError("动作后判断必须为matched/unmatched/uncertain或null。"))
    reject_if((action == "input_verified_text") != (isinstance(text, str) and bool(text)),
        CanonicalActionProtocolError("输入动作必须提供逐字目标正文；其他动作不携带正文。"))
    reject_if(action != "input_verified_text" and text is not None, CanonicalActionProtocolError("非输入动作不得携带正文。"))
    reject_if(action == "launch_app" and (not isinstance(app, str) or not app.strip()),
        CanonicalActionProtocolError("launch_app必须提供本地注册的App语义名称。"))
    reject_if(action != "launch_app" and app is not None, CanonicalActionProtocolError("非启动动作不得携带App参数。"))
    # Text actions consume the current input fact, not a model guess of an
    # internal projection ID. Extra IDs are raw diagnostics only.
    element_id = None if action in {"input_verified_text", "clear_verified_text", "press_enter"} else value.get("element_id")
    source_id = value.get("source_element_id")
    destination_id = value.get("destination_element_id")
    direction = value.get("direction")
    target = value.get("target")
    tap_point = value.get("tap_point")
    start = value.get("start")
    end = value.get("end")
    for name, part in {"action": action, "element_id": element_id, "source_element_id": source_id,
        "destination_element_id": destination_id, "direction": direction}.items():
        reject_if(part is not None and (not isinstance(part, str) or not part.strip()),
            CanonicalActionProtocolError(f"同响应decision.{name}必须是非空字符串或null。"))
    if status == "finish":
        reject_if(any(part is not None for part in (action, element_id, source_id, destination_id, direction, target))
            or tap_point is not None or start is not None or end is not None,
            CanonicalActionProtocolError("finish必须陈述当前截图完成事实且不得夹带动作。"))
    else:
        reject_if(action not in CANONICAL_ACTION_KINDS,
            CanonicalActionProtocolError("action必须是canonical动作。"))
        if action in MODEL_STEP_SINGLE_ELEMENT_ACTIONS:
            if action in MODEL_STEP_DIRECT_POINT_ACTIONS:
                reject_if(element_id is not None or any(part is not None for part in (
                    source_id, destination_id, direction)) or start is not None or end is not None,
                    CanonicalActionProtocolError("点按动作必须且只能使用decision.target绑定同帧唯一目标。"))
                target = _normalize_direct_target(target)
                _validate_model_point(tap_point, f"{action}.tap_point")
            else:
                if target is not None:
                    # Some Qwen responses include a plain description of the
                    # focused input alongside a text action.  It is
                    # diagnostic only: the binder always consumes the unique
                    # current-frame input audit and never this description as
                    # a selector.
                    _validate_text_target_description(target)
                    target = None
                reject_if(target is not None or any(part is not None for part in (
                    source_id, destination_id, direction)) or start is not None or end is not None,
                    CanonicalActionProtocolError("文字动作只消费同帧当前输入事实，不得夹带另一目标或轨迹。"))
                reject_if(tap_point is not None,
                    CanonicalActionProtocolError(f"{action}不得携带tap_point。"))
        elif action == "drag":
            reject_if(target is not None or element_id is not None or direction is not None or source_id is None
                or destination_id is None
                or source_id == destination_id or tap_point is not None or start is not None or end is not None,
                CanonicalActionProtocolError("drag必须且只能引用不同起点和终点。"))
        elif action == "scroll":
            reject_if(target is not None or source_id is not None or destination_id is not None
                or direction not in {"up", "down", "left", "right"}
                or tap_point is not None or start is not None or end is not None,
                CanonicalActionProtocolError("scroll必须声明唯一方向且不得携带自由轨迹。"))
        elif action == "swipe_element":
            reject_if(target is not None or element_id is None or source_id is not None or destination_id is not None
                or direction is not None or tap_point is not None or start is None or end is None,
                CanonicalActionProtocolError("swipe_element必须绑定一个元素以及起点和终点。"))
            _validate_model_point(start, "swipe_element.start")
            _validate_model_point(end, "swipe_element.end")
            reject_if(tuple(start) == tuple(end),
                CanonicalActionProtocolError("swipe_element起点和终点不能相同。"))
        else:
            if target is not None:
                # System actions have no model-selected surface target. Only
                # prose is redundant; execution payloads must never be hidden.
                _validate_system_target_description(target)
                target = None
            reject_if(any(part is not None for part in (element_id, source_id, destination_id, direction, target))
                or tap_point is not None or start is not None or end is not None,
                CanonicalActionProtocolError("系统动作不得携带元素或方向字段。"))
    normalized_reason = reason.strip() or ("" if status == "finish"
        else "当前截图选择一个推进目标的动作")
    return {"status": status, "action": action, "element_id": element_id,
        "source_element_id": source_id, "destination_element_id": destination_id, "direction": direction,
        "target": target, "tap_point": list(tap_point) if tap_point is not None else None,
        "start": list(start) if start is not None else None, "end": list(end) if end is not None else None,
        "confidence": float(confidence), "reason": normalized_reason, "text": text, "app": app,
        "previous_action_outcome": outcome, "state_action_consistent": state_action_consistent,
        "postcondition": postcondition}


def _validate_system_target_description(value: Any) -> None:
    reject_if(not isinstance(value, Mapping),
        CanonicalActionProtocolError("系统动作附带target只能是纯描述对象。"))
    reject_if(set(value) - _DIRECT_TARGET_FIELDS,
        CanonicalActionProtocolError("系统动作附带target不得包含几何、命令或其他动作字段。"))
    reject_if(any(part is not None and not isinstance(part, str)
        for key, part in value.items() if key != "evidence")
        or (value.get("evidence") is not None and (not isinstance(value["evidence"], list)
            or any(not isinstance(item, str) for item in value["evidence"]))),
        CanonicalActionProtocolError("系统动作附带target只能包含文字说明。"))


def _validate_text_target_description(value: Any) -> None:
    """Accept a redundant input description without letting it select a field."""
    reject_if(not isinstance(value, Mapping),
        CanonicalActionProtocolError("文字动作附带target只能是纯描述对象。"))
    reject_if(set(value) - _DIRECT_TARGET_FIELDS,
        CanonicalActionProtocolError("文字动作附带target不得包含几何、命令或其他动作字段。"))
    reject_if(any(part is not None and not isinstance(part, str)
        for key, part in value.items() if key != "evidence")
        or (value.get("evidence") is not None and (not isinstance(value["evidence"], list)
            or any(not isinstance(item, str) for item in value["evidence"]))),
        CanonicalActionProtocolError("文字动作附带target只能包含文字说明。"))


def _validate_model_point(value: Any, label: str) -> None:
    reject_if(not isinstance(value, (list, tuple)) or len(value) != 2
        or any(isinstance(part, bool) or not isinstance(part, (int, float)) for part in value),
        CanonicalActionProtocolError(f"{label}必须是两个数值。"))
    reject_if(any(not float(part) >= 0.0 or not float(part) < float("inf") for part in value),
        CanonicalActionProtocolError(f"{label}包含非法数值。"))


def _canonical_model_point(value: Any, label: str) -> tuple[float, float]:
    _validate_model_point(value, label)
    x, y = (float(part) for part in value)
    reject_if(not 0.0 <= x <= 1000.0 or not 0.0 <= y <= 1000.0,
        CanonicalActionProtocolError(f"{label}超出当前截图坐标范围。"))
    return (x / 1000.0, y / 1000.0)


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
            if element_id and self.action.action not in MODEL_STEP_DIRECT_POINT_ACTIONS:
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
    reject_if(kind not in CANONICAL_ACTION_KINDS,
        CanonicalActionProtocolError(f"Qwen 动作不属于 canonical 协议：{kind or 'missing'}。"))
    reject_if(kind not in available,
        CanonicalActionProtocolError(f"当前观察签发的动作集合不包含 Qwen 选择的 canonical 动作：{kind}。"))

    params: dict[str, Any] = {}
    if kind in MODEL_STEP_DIRECT_POINT_ACTIONS:
        target = _normalize_direct_target(payload.get("target"))
        params.update(_direct_target_params(target))
        if target["role"] == "input":
            element = _current_input_target(observation)
            params.update(_element_params(element))
        params["tap_point"] = _canonical_model_point(payload.get("tap_point"), f"{kind}.tap_point")
        if kind == "long_press":
            params["duration_ms"] = 800
    elif kind in {"input_verified_text", "clear_verified_text", "press_enter"}:
        element = _current_input_target(observation)
        params.update(_element_params(element))
        params.update(_text_action_params(kind, context=context, element=element,
            profile=text_transport_profile, text=payload.get("text")))
    elif kind == "drag":
        source = _selected_element(observation, payload.get("source_element_id"), kind)
        destination = _selected_element(observation, payload.get("destination_element_id"), kind)
        reject_if(source.element_id == destination.element_id,
            CanonicalActionProtocolError("drag 起点和终点不能相同。"))
        params.update(_element_params(source, prefix="source_"))
        params.update(_element_params(destination, prefix="destination_"))
    elif kind == "scroll":
        direction = str(payload.get("direction") or "").strip().lower()
        reject_if(direction not in {"up", "down", "left", "right"},
            CanonicalActionProtocolError("scroll 必须声明一个合法方向。"))
        params["direction"] = direction
        element_id = str(payload.get("element_id") or "").strip()
        if element_id:
            params.update(_element_params(_selected_element(observation, element_id, kind)))
    elif kind == "swipe_element":
        element = _selected_element(observation, payload.get("element_id"), kind)
        params.update(_element_params(element))
        params["start"] = _canonical_model_point(payload.get("start"), "swipe_element.start")
        params["end"] = _canonical_model_point(payload.get("end"), "swipe_element.end")
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


def _normalize_direct_target(value: Any) -> dict[str, Any]:
    reject_if(not isinstance(value, Mapping),
        CanonicalActionProtocolError("点按动作缺少严格decision.target对象。"))
    reject_if(set(value) - _DIRECT_TARGET_FIELDS,
        CanonicalActionProtocolError("decision.target不得携带几何或其他动作字段。"))
    diagnostic_id = value.get('element_id')
    result: dict[str, Any] = {'element_id': diagnostic_id.strip()
        if isinstance(diagnostic_id, str) and diagnostic_id.strip() else 'current_target'}
    for name in ("role", "meaning"):
        part = value.get(name)
        reject_if(not isinstance(part, str) or not part.strip(),
            CanonicalActionProtocolError(f"decision.target.{name}必须是非空字符串。"))
        result[name] = part.strip()
    label = value.get("label", "")
    result["label"] = label.strip() if isinstance(label, str) else ""
    raw_evidence = value.get("evidence", [])
    raw_evidence = raw_evidence if isinstance(raw_evidence, list) else []
    evidence = list(dict.fromkeys(item.strip() for item in raw_evidence
        if isinstance(item, str) and item.strip()))
    result["evidence"] = evidence
    return result


def _direct_target_params(target: Mapping[str, Any]) -> dict[str, Any]:
    return {"element_id": target["element_id"], "target": target["meaning"], "role": target["role"],
        "label": target["label"], "states": {}, "target_evidence": list(target["evidence"])}


def _current_input_target(observation: Any) -> UIElement:
    """Bind the selected input fact once; IDs never select a different field."""
    fields = [item for item in observation.scene.elements if item.role == "input"
        and item.states.get("input_field_id") == "current_input"]
    reject_if(len(fields) != 1,
        CanonicalActionProtocolError("同帧输入事实没有唯一可执行字段。"))
    element = fields[0]
    element.validate()
    return element


def _element_params(element: UIElement, *, prefix: str="") -> dict[str, Any]:
    return {f"{prefix}element_id": element.element_id, f"{prefix}target": element.meaning,
        f"{prefix}role": element.role, f"{prefix}label": element.label,
        f"{prefix}states": dict(element.states)}


def _text_action_params(kind: str, *, context: Any, element: UIElement,
    profile: TextTransportProfile | None, text: str | None) -> dict[str, Any]:
    reject_if(element.role != "input",
        CanonicalActionProtocolError("文字动作必须引用当前输入框。"))
    reject_if(profile is None or not profile.enabled,
        CanonicalActionProtocolError("当前设备未配置可用的 ADB Keyboard 文字通道。"))
    profile.validate()
    reject_if(profile.device_id != context.device_id,
        CanonicalActionProtocolError("ADB Keyboard profile 与当前设备不一致。"))
    reject_if(element.states.get("focused") is not True,
        CanonicalActionProtocolError("文字动作必须引用当前画面明确已聚焦的输入框。"))
    prior = element.states.get("value")
    reject_if(not isinstance(prior, str), CanonicalActionProtocolError("文字动作缺少当前输入值。"))
    field_id = str(element.states.get("input_field_id") or "").strip()
    reject_if(field_id != "current_input", CanonicalActionProtocolError("文字动作没有绑定当前输入事实。"))
    params = {"text_transport": "adb_keyboard", "input_field_id": field_id, "prior_input_value": prior}
    if kind == "clear_verified_text":
        reject_if("clear_text" not in profile.capabilities,
            CanonicalActionProtocolError("当前 ADB Keyboard 不能唯一清空该 typed 字段。"))
        reject_if(not prior and not str(element.states.get("ime_preedit_text") or ""),
            CanonicalActionProtocolError("当前输入框已经为空，不得重复清空。"))
        return {**params, "expected_input_value": ""}
    if kind == "press_enter":
        reject_if(element.states.get("input_multiline") is not True,
            CanonicalActionProtocolError("换行必须绑定当前 typed 多行字段。"))
        target = prior + "\n"
    else:
        target = text
        reject_if(not isinstance(target, str) or not target, CanonicalActionProtocolError("输入缺少目标正文。"))
        reject_if(context.exact_input_text is not None and target != context.exact_input_text,
            CanonicalActionProtocolError("输入正文与用户明确提供的逐字正文不同。"))
    reject_if("append_text" not in profile.capabilities or not target.startswith(prior) or target == prior,
        CanonicalActionProtocolError("当前 ADB Keyboard 不能建立唯一 prior/fragment/expected 事务。"))
    return {**params, "text": target, "input_fragment": target[len(prior):], "expected_input_value": target}


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
