from __future__ import annotations
import json
import re
import unittest
from PIL import Image
from agent.infrastructure.generic_scene_observer import (
    INPUT_STRUCTURE_AUDIT_VERSION,
    SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
)
from agent.domain.ui_scene import UI_SCENE_PROTOCOL_VERSION, UIScene


def current_axis_grid_payload(value: dict, *, request_height: int) -> dict:
    """Translate old normalized fixtures to the sole current wire contract."""

    payload = json.loads(json.dumps(value, ensure_ascii=False))
    flat_scene = payload.get("protocol_version") == UI_SCENE_PROTOCOL_VERSION
    envelope = isinstance(payload.get("scene"), dict) and isinstance(payload.get("decision"), dict)
    if envelope:
        old_height = (payload.get("coordinate_space") or {}).get("height")
        if isinstance(old_height, (int, float)) and old_height not in (0, 1000):
            decision = payload.get("decision")
            point = decision.get("tap_point") if isinstance(decision, dict) else None
            if isinstance(point, list) and len(point) == 2 and isinstance(point[1], (int, float)):
                point[1] = round(point[1] * 1000 / old_height)
            scene = payload.get("scene")
            for element in scene.get("elements", []) if isinstance(scene, dict) else []:
                bounds = element.get("bounds") if isinstance(element, dict) else None
                if isinstance(bounds, list) and len(bounds) == 4:
                    bounds[1] = round(bounds[1] * 1000 / old_height)
                    bounds[3] = round(bounds[3] * 1000 / old_height)
        if "protocol_version" not in payload:
            payload["protocol_version"] = SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION
    if flat_scene:
        payload = {
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {
                "kind": "axis_grid",
                "width": 1000,
                "height": request_height,
            },
            "scene": payload,
            "input_structure": None,
        }
    if not flat_scene:
        if (
            payload.get("protocol_version")
            == SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION
            and "decision" not in payload
        ):
            payload["decision"] = default_model_decision()
        if payload.get("protocol_version") == SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION:
            payload["coordinate_space"] = {
                "kind": "axis_grid",
                "width": 1000,
                "height": 1000,
            }
            decision = payload.get("decision")
            if isinstance(decision, dict) and decision.get("action") and "postcondition" not in decision:
                decision["postcondition"] = {
                    "status": "unknown",
                    "fact": "当前截图无法确认上一步动作结果",
                }
        if set(payload) >= {"coordinate_space", "tap_point"} and isinstance(payload.get("coordinate_space"), dict):
            payload["coordinate_space"] = {"kind": "axis_grid", "width": 1000, "height": request_height}
        _adapt_direct_point_fixture(payload)
        return payload
    payload.setdefault("decision", default_model_decision())
    _adapt_direct_point_fixture(payload)
    return payload


def _adapt_direct_point_fixture(payload: dict) -> None:
    """Project historical point fixtures into the strict target-only branch."""

    decision = payload.get("decision")
    scene = payload.get("scene")
    if (not isinstance(decision, dict) or not isinstance(scene, dict)
        or decision.get("action") not in {"tap_semantic", "dismiss_overlay", "press_enter", "double_tap",
            "long_press"} or isinstance(decision.get("target"), dict)):
        return
    element_id = str(decision.pop("element_id", "") or "").strip()
    elements = scene.get("elements") if isinstance(scene.get("elements"), list) else []
    selected = next((item for item in elements if isinstance(item, dict)
        and str(item.get("element_id") or "").strip() == element_id), {})
    role = str(selected.get("role") or "button").strip()
    meaning = str(selected.get("meaning") or "test_target").strip()
    label = str(selected.get("label") or "").strip()
    evidence = [str(item).strip() for item in selected.get("evidence") or []
        if isinstance(item, str) and item.strip()]
    decision["target"] = {"element_id": element_id or "test-direct-target", "role": role,
        "meaning": meaning, "label": label, "evidence": evidence or [label or meaning]}
    scene["elements"] = []


def default_model_decision() -> dict:
    """Give legacy scene fixtures a neutral decision under the current wire contract."""

    return {
        "status": "finish",
        "action": None,
        "element_id": None,
        "source_element_id": None,
        "destination_element_id": None,
        "direction": None,
        "evidence_refs": ["scene.summary"],
        "confidence": 0.95,
        "reason": "test fixture only: current scene is the requested observation",
        "postcondition": None,
        "previous_action_outcome": None,
    }


def request_axis_height(messages: list[dict]) -> int:
    prompt = json.dumps(messages, ensure_ascii=False)
    match = re.search(
        r"coordinate_space.{0,40}?axis_grid.{0,40}?width.{0,15}?1000" r".{0,40}?height.{0,15}?(\d+)",
        prompt,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError("single-step prompt omitted the current axis grid")
    return int(match.group(1))


class FakeProvider:
    configured = True

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def status(self) -> dict:
        return {"configured": True, "model": "fake-qwen"}

    def _chat(
        self,
        messages: list[dict],
        max_tokens: int | None,
        *,
        timeout: float | None = None,
        max_attempts: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        self.calls += 1
        self.messages = messages
        self.max_tokens = max_tokens
        self.call_options = {
            "timeout": timeout,
            "max_attempts": max_attempts,
            "response_format": response_format,
        }
        payload = current_axis_grid_payload(
            self.payload,
            request_height=request_axis_height(messages),
        )
        return json.dumps(payload, ensure_ascii=False)


class SequenceProvider(FakeProvider):
    def __init__(self, responses: list[str | dict | BaseException]) -> None:
        super().__init__({})
        self.responses = list(responses)
        self.max_tokens_seen: list[int] = []
        self.messages_seen: list[list[dict]] = []

    def _chat(
        self,
        messages: list[dict],
        max_tokens: int | None,
        *,
        timeout: float | None = None,
        max_attempts: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        self.calls += 1
        self.messages_seen.append(messages)
        self.max_tokens_seen.append(max_tokens)
        self.call_options = {
            "timeout": timeout,
            "max_attempts": max_attempts,
            "response_format": response_format,
        }
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        if isinstance(value, dict):
            value = current_axis_grid_payload(
                value,
                request_height=request_axis_height(messages),
            )
        elif isinstance(value, str):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                decoded = None
            if (
                isinstance(decoded, dict)
                and decoded.get("protocol_version")
                == SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION
                and "decision" not in decoded
            ):
                decoded["decision"] = default_model_decision()
            if (isinstance(decoded, dict) and decoded.get("protocol_version")
                == SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION):
                _adapt_direct_point_fixture(decoded)
                value = json.dumps(decoded, ensure_ascii=False)
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


class _BaseStructuredDecisionContractTests(unittest.TestCase):
    @staticmethod
    def _context(state: str) -> dict:
        return {"entities": {"active_subgoal_visual_context": {
            "subgoal_id": "input_message",
            "objective": "在当前输入框输入指定文字",
            "constraints": [],
            "completion_conditions": ["输入框正文达到目标值"],
            "execution_class": "input",
            "goal_entities": {"active_input_field_id": "primary_input",
                "active_input_field_label": "消息", "active_input_multiline": False,
                "active_input_transaction_text": "sample"},
            "transition_receipt": {"state": state, "subgoal_id": "input_message",
                "required_operation": "input_verified_text", "executed_operation": ""
                if state == "pending" else "input_verified_text", "effect_ids": []},
        }}}


def stable_frames(color: tuple[int, int, int] = (30, 40, 50)) -> list[Image.Image]:
    return [Image.new("RGB", (540, 960), color) for _ in range(4)]


def unique_goal_element(scene: UIScene):
    """Test-only inspection; production actions use the Qwen-selected element ID."""

    matches = tuple(item for item in scene.elements if item.states.get("goal_relevant") is True)
    return matches[0] if len(matches) == 1 else None










def scene_payload() -> dict:
    return {
        "protocol_version": UI_SCENE_PROTOCOL_VERSION,
        "foreground_app_id": "calculator",
        "screen_id": "app_home",
        "summary": "计算器首页",
        "system_ui": {
            "immersive_or_fullscreen": False,
            "navigation_bar_visible": True,
        },
        "camera_alignment": {
            "camera_layout_orientation": "portrait",
            "phone_content_rotation": "upright",
            "confidence": 0.95,
            "evidence": ["手机界面文字在原始相机画布中正向显示"],
        },
        "elements": [
            {
                "element_id": "e1",
                "role": "button",
                "meaning": "digit_key",
                "label": "7",
                "bounds": [100, 600, 260, 760],
                "confidence": 0.98,
                "states": {"enabled": True},
                "evidence": ["7"],
            }
        ],
        "overlays": [],
        "stable": True,
        "confidence": 0.96,
        "fingerprint": "model-value-must-not-be-trusted",
    }


def input_audit_payload(
    *,
    application_inputs: list[dict] | None = None,
    ime_preedit_regions: list[dict] | None = None,
    keyboard: dict | None = None,
) -> dict:
    resolved_keyboard = dict(
        keyboard
        or {
            "visible": False,
            "bounds": None,
            "layout": "unknown",
            "input_mode": "unknown",
            "mode_switch": None,
        }
    )
    resolved_keyboard.setdefault("case_mode", "unknown")
    resolved_keyboard.setdefault("case_switch", None)
    resolved_keyboard.setdefault("literal_keys", [])
    resolved_keyboard.setdefault("layout_switches", [])
    if (
        resolved_keyboard.get("visible") is True
        and str(resolved_keyboard.get("layout") or "").strip().casefold() == "qwerty"
        and resolved_keyboard.get("input_mode") == "direct_latin"
        and "qwerty_anchors" not in resolved_keyboard
    ):
        resolved_keyboard["qwerty_anchors"] = {
            "q": [115, 704],
            "p": [875, 704],
            "a": [157, 773],
            "l": [832, 773],
            "z": [241, 844],
            "m": [747, 844],
            "backspace": [875, 844],
        }
    return {
        "protocol_version": INPUT_STRUCTURE_AUDIT_VERSION,
        "application_inputs": list(application_inputs or []),
        "ime_preedit_regions": list(ime_preedit_regions or []),
        "keyboard": resolved_keyboard,
    }






def audited_application_input(
    *,
    structure_id: str = "field",
    bounds: list[int] | None = None,
    fully_visible: bool = True,
    text: str = "已有文字",
    preedit_text: str = "",
    placeholder: str = "",
    confidence: float = 0.98,
    right_button: dict | None = None,
    field_labels: list[str] | None = None,
    visible_editable_cues: list[str] | None = None,
    caret_line_index: int | None = None,
    focused: bool | None = True,
) -> dict:
    value = {
        "structure_id": structure_id,
        "bounds": bounds or [110, 40, 850, 110],
        "fully_visible": fully_visible,
        "text": text,
        "preedit_text": preedit_text,
        "placeholder": placeholder,
        "visible_editable_cues": list(["完整横向输入边框"] if visible_editable_cues is None else visible_editable_cues),
        "caret_line_index": caret_line_index,
        "focused": focused,
        "confidence": confidence,
        "right_button": right_button,
    }
    if field_labels is not None:
        value["field_labels"] = list(field_labels)
    return value


def audited_text_input_scene(
    bounds: list[int],
) -> tuple[dict, dict]:
    scene = scene_payload()
    scene.update(
        {
            "foreground_app_id": "com.example.messaging",
            "screen_id": "named_conversation",
            "summary": "指定会话页底部有一个空输入框",
            "elements": [
                {
                    "element_id": "message-input",
                    "role": "input",
                    "meaning": "message_input_field",
                    "label": "",
                    "bounds": list(bounds),
                    "confidence": 1.0,
                    "states": {
                        "goal_relevant": True,
                        "fully_visible": True,
                        "value": "",
                    },
                    "evidence": ["底部唯一完整白色输入区域"],
                }
            ],
        }
    )
    audit = input_audit_payload(
        application_inputs=[
            audited_application_input(
                structure_id="message",
                bounds=list(bounds),
                text="",
                placeholder="",
                visible_editable_cues=["白色矩形背景"],
            )
        ]
    )
    return scene, audit
