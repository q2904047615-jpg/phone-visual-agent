from __future__ import annotations
import json
import re
import unittest
import time
from unittest.mock import patch
from PIL import Image, ImageDraw, ImageFilter
from agent.infrastructure.generic_scene_observer import (
    INPUT_STRUCTURE_AUDIT_VERSION,
    SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
    _single_step_response_format,
)
from agent.domain.ui_scene import UI_SCENE_PROTOCOL_VERSION, UIScene
from agent.infrastructure.dashscope_vision_provider import _extract_json_object
from agent.domain.vision_model import VisionAgentError
from agent.domain.visual_evidence import VisualObstruction


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
        if self.calls == 2 and payload.get("protocol_version") == UI_SCENE_PROTOCOL_VERSION:
            payload = targeted_delta_payload(
                elements=payload.get("elements") or [],
                summary_addendum=str(payload.get("summary") or "")[:120],
                confidence=float(payload.get("confidence") or 0.0),
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
        # Existing scene fixtures describe the intended refined facts. Adapt
        # only the second model response to the production targeted-delta wire
        # contract so the large historical suite does not duplicate fixtures.
        if self.calls == 2 and isinstance(value, dict) and value.get("protocol_version") == UI_SCENE_PROTOCOL_VERSION:
            value = targeted_delta_payload(
                elements=value.get("elements") or [],
                summary_addendum=str(value.get("summary") or "")[:120],
                confidence=float(value.get("confidence") or 0.0),
            )
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


def icon_cluster_frames() -> list[Image.Image]:
    image = Image.new("RGB", (540, 960), (30, 40, 50))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 539, 96), fill=(215, 215, 215))
    draw.rounded_rectangle((400, 18, 430, 53), radius=4, outline=(35, 35, 35), width=3)
    draw.line((415, 27, 415, 44), fill=(35, 35, 35), width=3)
    draw.line((407, 35, 423, 35), fill=(35, 35, 35), width=3)
    draw.arc((444, 19, 474, 50), 35, 330, fill=(35, 35, 35), width=4)
    draw.polygon(((468, 18), (477, 22), (468, 28)), fill=(35, 35, 35))
    return [image.copy() for _ in range(4)]


def stable_frames_with_one_sharp_center() -> list[Image.Image]:
    base = Image.new("RGB", (540, 960), (30, 40, 50))
    checker = Image.new("RGB", (280, 420), (10, 10, 10))
    pixels = checker.load()
    for y in range(checker.height):
        for x in range(checker.width):
            value = 240 if ((x // 8) + (y // 8)) % 2 else 10
            pixels[x, y] = (value, value, value)
    sharp = base.copy()
    sharp.paste(checker, (130, 270))
    soft_center = checker.filter(ImageFilter.GaussianBlur(radius=6.0))
    soft = base.copy()
    soft.paste(soft_center, (130, 270))
    return [soft.copy(), sharp, soft.copy(), soft.copy()]


def converged_frames_with_sharp_stale_leader() -> list[Image.Image]:
    settled = Image.new("RGB", (540, 960), (30, 40, 50))
    stale = settled.copy()
    draw = ImageDraw.Draw(stale)
    for y in range(0, 960, 8):
        draw.line((0, y, 539, y), fill="white" if (y // 8) % 2 else "black", width=4)
    return [stale, settled.copy(), settled.copy(), settled.copy()]


def frames_with_top_obstruction() -> list[Image.Image]:
    image = Image.new("RGB", (540, 960), "#dddddd")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 300, 44), fill="black")
    draw.rectangle((14, 12, 175, 21), fill="white")
    return [image.copy() for _ in range(4)]


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


def app_identity_audit_payload(
    app_id: str,
    *,
    confidence: float = 0.98,
    evidence: list[str] | None = None,
) -> dict:
    return {
        "protocol_version": FOREGROUND_APP_IDENTITY_AUDIT_VERSION,
        "foreground_app_id": app_id,
        "confidence": confidence,
        "evidence": list(evidence or ["前台应用视觉身份清晰可辨"]),
    }


def extra_brace_scene_response() -> str:
    payload = scene_payload()
    payload["foreground_app_id"] = "launcher"
    payload["elements"][0]["meaning"] = "browser"
    payload["elements"][0]["label"] = "浏览器"
    payload["elements"][0]["evidence"] = ["浏览器"]
    valid = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    malformed = valid.replace('}],"overlays"', '}}],"overlays"', 1)
    if malformed == valid:
        raise AssertionError("测试响应未插入额外右花括号。")
    return malformed


def targeted_delta_payload(
    *,
    elements: list[dict] | None = None,
    summary_addendum: str = "",
    confidence: float = 0.96,
) -> dict:
    return {
        "protocol_version": TARGETED_SCENE_DELTA_PROTOCOL_VERSION,
        "summary_addendum": summary_addendum,
        "elements": list(elements or []),
        "confidence": confidence,
    }


def extra_brace_targeted_delta_response() -> str:
    payload = targeted_delta_payload(elements=scene_payload()["elements"])
    valid = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    malformed = valid.replace('}],"confidence"', '}}],"confidence"', 1)
    if malformed == valid:
        raise AssertionError("测试响应未插入额外右花括号。")
    return malformed


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


def keyboard_routing_scene(value: str, fingerprint: str) -> UIScene:
    payload = scene_payload()
    payload.update({
        "foreground_app_id": "sample.app",
        "screen_id": "editor",
        "summary": "唯一输入框和键盘可见",
        "elements": [{
            "element_id": "input-1",
            "role": "input",
            "meaning": "application_text_input",
            "bounds": [0.13, 0.54, 0.69, 0.61],
            "confidence": 1.0,
            "label": value,
            "states": {
                "goal_relevant": False,
                "fully_visible": True,
                "focused": True,
                "value": value,
            },
            "evidence": [f"应用输入框当前文字：{value}"],
        }],
        "overlays": ["软键盘可见"],
        "fingerprint": fingerprint,
    })
    return UIScene.from_dict(payload)


def keyboard_routing_audit(
    *,
    value: str,
    layout: str,
    input_mode: str,
    literal: str | None = None,
    mode_switch: dict | None = None,
    layout_switches: list[dict] | None = None,
) -> str:
    qwerty_anchors = None
    if layout == "qwerty":
        qwerty_anchors = {
            "q": [122, 710],
            "p": [880, 710],
            "a": [164, 782],
            "l": [838, 782],
            "z": [248, 853],
            "m": [754, 853],
            "backspace": [880, 853],
        }
    return json.dumps(input_audit_payload(
        application_inputs=[audited_application_input(
            structure_id="app-input-1",
            bounds=[130, 540, 690, 610],
            text=value,
            visible_editable_cues=["cursor"],
        )],
        keyboard={
            "visible": True,
            "bounds": [0, 660, 1000, 1000],
            "layout": layout,
            "input_mode": input_mode,
            "case_mode": (
                "lower"
                if layout == "qwerty" and input_mode == "direct_latin"
                else "unknown"
            ),
            "qwerty_anchors": qwerty_anchors,
            "mode_switch": mode_switch,
            "backspace_key": None,
            "enter_key": None,
            "case_switch": None,
            "literal_keys": ([] if literal is None else [{
                "value": literal,
                "label": literal,
                "key_kind": "character",
                "bounds": [420, 710, 580, 780],
                "confidence": 1.0,
                "fully_visible": True,
            }]),
            "layout_switches": list(layout_switches or []),
        },
    ), ensure_ascii=False)


def icon_cluster_audit_payload(
    *,
    controls: list[dict] | None = None,
    cluster_complete: bool = True,
    cluster_bounds: list[int] | None = None,
) -> dict:
    resolved_controls = controls
    if resolved_controls is None:
        resolved_controls = [
            {
                "control_id": "reload-control",
                "semantic_class": "reload",
                "bounds": [820, 20, 875, 75],
                "confidence": 0.96,
                "fully_visible": True,
                "single_glyph": True,
                "shape_cues": ["curved_arc", "arrowhead"],
            },
            {
                "control_id": "bookmark-control",
                "semantic_class": "bookmark",
                "bounds": [740, 20, 795, 75],
                "confidence": 0.97,
                "fully_visible": True,
                "single_glyph": True,
                "shape_cues": ["bookmark_outline"],
            },
        ]
    return {
        "protocol_version": ICON_CLUSTER_AUDIT_VERSION,
        "cluster_complete": cluster_complete,
        "cluster_bounds": (
            cluster_bounds if cluster_bounds is not None else [700, 0, 920, 100] if resolved_controls else None
        ),
        "controls": resolved_controls,
    }


def localized_icon_cluster_audit_payload() -> dict:
    """Default audit expressed in the padded [645,0,975,125] crop."""

    return icon_cluster_audit_payload(
        controls=[
            {
                "control_id": "reload-control-local",
                "semantic_class": "reload",
                "bounds": [530, 160, 697, 600],
                "confidence": 0.97,
                "fully_visible": True,
                "single_glyph": True,
                "shape_cues": ["curved_arc", "arrowhead"],
            },
            {
                "control_id": "bookmark-control-local",
                "semantic_class": "bookmark",
                "bounds": [288, 160, 455, 600],
                "confidence": 0.97,
                "fully_visible": True,
                "single_glyph": True,
                "shape_cues": ["bookmark_outline"],
            },
        ],
        cluster_bounds=[167, 0, 833, 800],
    )


def system_ui_audit_payload(
    *,
    immersive_or_fullscreen: bool | str = True,
    navigation_bar_visible: bool | str = False,
    confidence: float = 0.95,
    evidence: list[object] | None = None,
) -> dict:
    return {
        "protocol_version": SYSTEM_UI_AUDIT_VERSION,
        "immersive_or_fullscreen": immersive_or_fullscreen,
        "navigation_bar_visible": navigation_bar_visible,
        "confidence": confidence,
        "evidence": evidence or ["App内容填满手机显示区域", "系统导航栏未显示"],
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


MULTIFIELD_FIELDS = [
    {"field_id": "subject_field", "field_label": "主题", "text": "first"},
    {"field_id": "body_field", "field_label": "正文", "text": "second"},
]


def multifield_next_context(fields: list[dict], *, dependency: bool = True) -> dict:
    by_id = {item["field_id"]: item for item in fields}
    source, target = by_id["subject_field"], by_id["body_field"]
    markers = {
        "active_input_transaction_text": target["text"],
        "active_input_field_id": target["field_id"],
        "active_input_field_label": target["field_label"],
        "active_input_multiline": False,
    }
    if dependency:
        markers.update(
            {
                "active_input_predecessor_field_id": source["field_id"],
                "active_input_predecessor_field_label": source["field_label"],
                "active_input_predecessor_text": source["text"],
            }
        )
    return {
        "entities": {
            "input_fields": fields,
            "active_subgoal_visual_context": {
                "subgoal_id": "input_body",
                "objective": "输入下一字段",
                "constraints": [],
                "completion_conditions": [],
                "execution_class": "navigate",
                "goal_entities": markers,
            },
        }
    }


def multifield_next_base(fields: list[dict], *, duplicate: bool = False):
    source = next(item for item in fields if item["field_id"] == "subject_field")
    visible = {
        "element_id": "source-visible",
        "role": "input",
        "meaning": "application_text_input",
        "label": source["field_label"],
        "bounds": [120, 430, 880, 590],
        "confidence": 0.98,
        "states": {
            "goal_relevant": True,
            "fully_visible": True,
            "focused": True,
            "value": source["text"],
            "input_field_id": source["field_id"],
            "input_field_label": source["field_label"],
        },
        "evidence": [source["field_label"], "caret"],
    }
    payload = scene_payload()
    payload["elements"] = [visible]
    if duplicate:
        payload["elements"].append(
            {
                **visible,
                "element_id": "source-duplicate",
                "bounds": [120, 250, 880, 400],
            }
        )
    return _parse_scene(json.dumps(payload, ensure_ascii=False), fingerprint="f" * 64)


def multifield_next_audit(
    fields: list[dict],
    *,
    action: str = "next",
    fully_visible: bool = True,
    confidence: float = 0.98,
    duplicate: bool = False,
) -> dict:
    source = next(item for item in fields if item["field_id"] == "subject_field")
    application_inputs = [
        audited_application_input(
            structure_id="source",
            bounds=[120, 430, 880, 590],
            text=source["text"],
            field_labels=[source["field_label"]],
        )
    ]
    if duplicate:
        application_inputs.append(
            audited_application_input(
                structure_id="source-duplicate",
                bounds=[120, 250, 880, 400],
                text=source["text"],
                field_labels=[source["field_label"]],
            )
        )
    return input_audit_payload(
        application_inputs=application_inputs,
        keyboard={
            "visible": True,
            "bounds": [0, 600, 1000, 1000],
            "layout": "qwerty",
            "input_mode": "direct_latin",
            "case_mode": "lower",
            "mode_switch": None,
            "enter_key": {
                "label": "下一步",
                "bounds": [820, 920, 990, 985],
                "confidence": confidence,
                "fully_visible": fully_visible,
                "key_action": action,
            },
        },
    )
