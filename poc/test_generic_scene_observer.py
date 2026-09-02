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
    SingleStepGenericSceneObserver,
    _apply_input_structure_audit,
)
from agent.domain.ui_scene import UI_SCENE_PROTOCOL_VERSION, UIScene
from agent.infrastructure.dashscope_vision_provider import _image_data_url
from agent.domain.vision_model import VisionAgentError
from agent.domain.foreground_app_identity import ForegroundAppIdentity
from agent.domain.visual_evidence import VisualObstruction


def current_axis_grid_payload(value: dict, *, request_height: int) -> dict:
    """Translate old normalized fixtures to the sole current wire contract."""

    payload = json.loads(json.dumps(value, ensure_ascii=False))
    flat_scene = payload.get("protocol_version") == UI_SCENE_PROTOCOL_VERSION
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
    coordinate_space = payload.get("coordinate_space")
    normalized_fixture = coordinate_space == {
        "kind": "normalized_1000",
        "width": 1000,
        "height": 1000,
    }
    if not (flat_scene or normalized_fixture):
        if (
            payload.get("protocol_version")
            == SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION
            and "decision" not in payload
        ):
            payload["decision"] = default_model_decision()
        return payload
    if normalized_fixture:
        payload["coordinate_space"] = {
            "kind": "axis_grid",
            "width": 1000,
            "height": request_height,
        }

    def scale(node, *, anchors: bool = False) -> None:
        if isinstance(node, list):
            for item in node:
                scale(item, anchors=anchors)
        elif isinstance(node, dict):
            for key, item in node.items():
                if key == "bounds" and isinstance(item, list) and len(item) == 4:
                    item[1] = round(item[1] * request_height / 1000)
                    item[3] = round(item[3] * request_height / 1000)
                elif anchors and isinstance(item, list) and len(item) == 2:
                    item[1] = round(item[1] * request_height / 1000)
                else:
                    scale(item, anchors=(key == "qwerty_anchors"))

    scale(payload.get("scene"))
    scale(payload.get("input_structure"))
    payload.setdefault("decision", default_model_decision())
    return payload


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
        max_tokens: int,
        *,
        timeout: float | None = None,
        max_attempts: int | None = None,
        response_format: dict[str, str] | None = None,
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
        max_tokens: int,
        *,
        timeout: float | None = None,
        max_attempts: int | None = None,
        response_format: dict[str, str] | None = None,
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


class TrustedForegroundIdentityTests(unittest.TestCase):
    def test_signed_system_identity_overrides_model_app_id_but_not_page_semantics(self) -> None:
        provider = FakeProvider(scene_payload())
        now = time.time()
        identity = ForegroundAppIdentity(device_id="device-local-01", package_name="com.android.settings",
            source="usage_stats", event_at_epoch=now - 10, observed_at_epoch=now)

        observed = SingleStepGenericSceneObserver(provider).observe(frames=stable_frames(),
            goal_context={"objective": "打开系统设置"}, device_id="device-local-01",
            available_action_kinds={"home", "launch_app"}, trusted_foreground_identity=identity)

        self.assertEqual("com.android.settings", observed.foreground_app_id)
        self.assertEqual("app_home", observed.screen_id)
        prompt = provider.messages[1]["content"][0]["text"]
        self.assertIn("com.android.settings", prompt)
        self.assertIn("不得根据JPEG", prompt)

    def test_no_system_identity_keeps_visual_fallback_and_system_identity_overrides_next_observation(self) -> None:
        provider = FakeProvider(scene_payload())
        observer = SingleStepGenericSceneObserver(provider)
        frames = stable_frames()
        visual = observer.observe(frames=frames, goal_context={"objective": "观察当前页面"},
            device_id="device-local-01", available_action_kinds={"home"})
        now = time.time()
        identity = ForegroundAppIdentity(device_id="device-local-01", package_name="com.tencent.mm",
            source="editor_info", event_at_epoch=now, observed_at_epoch=now)
        system = observer.observe(frames=frames, goal_context={"objective": "观察当前页面"},
            device_id="device-local-01", available_action_kinds={"home"},
            trusted_foreground_identity=identity)

        self.assertEqual("calculator", visual.foreground_app_id)
        self.assertEqual("com.tencent.mm", system.foreground_app_id)
        self.assertEqual(2, provider.calls)


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
    placeholder: str = "",
    confidence: float = 0.98,
    right_button: dict | None = None,
    field_labels: list[str] | None = None,
    visible_editable_cues: list[str] | None = None,
    caret_line_index: int | None = None,
) -> dict:
    value = {
        "structure_id": structure_id,
        "bounds": bounds or [110, 40, 850, 110],
        "fully_visible": fully_visible,
        "text": text,
        "placeholder": placeholder,
        "visible_editable_cues": list(["完整横向输入边框"] if visible_editable_cues is None else visible_editable_cues),
        "caret_line_index": caret_line_index,
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


class SingleStepGenericSceneObserverTests(unittest.TestCase):
    def test_obsolete_blocked_decision_is_rejected_by_wire_contract(self) -> None:
        payload = {
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {
                "kind": "normalized_1000",
                "width": 1000,
                "height": 1000,
            },
            "scene": scene_payload(),
            "input_structure": None,
            "decision": {
                "status": "blocked",
                "action": None,
                "element_id": None,
                "source_element_id": None,
                "destination_element_id": None,
                "direction": None,
                "evidence_refs": [],
                "confidence": 0.9,
                "reason": "legacy model veto",
            },
        }
        provider = SequenceProvider([payload])

        with self.assertRaisesRegex(VisionAgentError, "只允许action或finish"):
            SingleStepGenericSceneObserver(provider).observe(
                frames=stable_frames(),
                goal_context={"objective": "返回手机主屏幕"},
                device_id="device-local-01",
                available_action_kinds={"home"},
            )

    def test_missing_fixed_outer_protocol_version_is_filled_once(self) -> None:
        payload = {
            "coordinate_space": {
                "kind": "normalized_1000",
                "width": 1000,
                "height": 1000,
            },
            "scene": scene_payload(),
            "input_structure": None,
            "decision": {"status": "finish", "evidence_refs": ["scene.summary"]},
        }
        provider = SequenceProvider([payload])

        observed = SingleStepGenericSceneObserver(provider).observe(
            frames=stable_frames(),
            goal_context={"objective": "确认当前页面"},
            device_id="device-local-01",
        )

        self.assertEqual("app_home", observed.screen_id)
        self.assertEqual(1, provider.calls)

    def test_explicit_conflicting_outer_protocol_version_is_rejected(self) -> None:
        payload = {
            "protocol_version": "conflicting-version",
            "coordinate_space": {
                "kind": "normalized_1000",
                "width": 1000,
                "height": 1000,
            },
            "scene": scene_payload(),
            "input_structure": None,
            "decision": {"status": "finish", "evidence_refs": ["scene.summary"]},
        }
        provider = SequenceProvider([payload])

        with self.assertRaisesRegex(VisionAgentError, "协议版本不匹配"):
            SingleStepGenericSceneObserver(provider).observe(
                frames=stable_frames(),
                goal_context={"objective": "确认当前页面"},
                device_id="device-local-01",
            )

        self.assertEqual(1, provider.calls)

    def test_nested_action_injection_inside_harmless_outer_metadata_is_rejected(self) -> None:
        payload = {
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "normalized_1000", "width": 1000, "height": 1000},
            "scene": scene_payload(),
            "input_structure": None,
            "decision": {"status": "finish", "evidence_refs": ["scene.summary"]},
            "metadata": {"plan": ["click once", "click again"]},
        }
        provider = SequenceProvider([payload])

        with self.assertRaisesRegex(VisionAgentError, "动作或计划"):
            SingleStepGenericSceneObserver(provider).observe(
                frames=stable_frames(), goal_context={"objective": "确认当前页面"},
                device_id="device-local-01")

        self.assertEqual(1, provider.calls)

    def test_missing_optional_system_ui_and_camera_use_local_frame_facts(self) -> None:
        compact_scene = scene_payload()
        compact_scene.pop("system_ui")
        compact_scene.pop("camera_alignment")
        provider = SequenceProvider(
            [
                {
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    "coordinate_space": {
                        "kind": "normalized_1000",
                        "width": 1000,
                        "height": 1000,
                    },
                    "scene": compact_scene,
                    "input_structure": None,
                    "decision": {
                        "status": "finish",
                        "evidence_refs": ["scene.summary"],
                    },
                }
            ]
        )

        observed = SingleStepGenericSceneObserver(provider).observe(
            frames=stable_frames(),
            goal_context={"objective": "确认当前页面"},
            device_id="device-local-01",
        )

        self.assertEqual("unknown", observed.system_ui.navigation_bar_visible)
        self.assertEqual("portrait", observed.camera_alignment.camera_layout_orientation)
        self.assertEqual("unknown", observed.camera_alignment.phone_content_rotation)

    def test_duplicate_finish_evidence_is_normalized_without_a_second_veto(self) -> None:
        provider = SequenceProvider(
            [
                {
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    "coordinate_space": {
                        "kind": "normalized_1000",
                        "width": 1000,
                        "height": 1000,
                    },
                    "scene": scene_payload(),
                    "input_structure": None,
                    "decision": {
                        "status": "finish",
                        "evidence_refs": ["scene.summary", "scene.summary"],
                    },
                }
            ]
        )
        observer = SingleStepGenericSceneObserver(provider)

        observed, model_decision = observer.observe_with_decision(
            frames=stable_frames(),
            goal_context={"objective": "确认当前页面"},
            device_id="device-local-01",
        )

        self.assertEqual(
            ["scene.summary"],
            model_decision["evidence_refs"],
        )

    def test_explicit_system_home_observation_sends_unmasked_phone_frame(
        self,
    ) -> None:
        cases = (
            (
                "chinese",
                (173, 61, 211),
                {
                    "objective": "回到手机主屏幕",
                    "execution_class": "navigate",
                    "completion_conditions": ["手机主屏幕可见"],
                },
            ),
            (
                "english-variation",
                (27, 189, 116),
                {
                    "objective": "return to the phone home screen",
                    "execution_class": "navigate",
                    "completion_conditions": ["phone home screen is visible"],
                },
            ),
        )
        for name, color, goal_context in cases:
            with self.subTest(name=name):
                envelope = {
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    "coordinate_space": {
                        "kind": "normalized_1000",
                        "width": 1000,
                        "height": 1000,
                    },
                    "scene": scene_payload(),
                    "input_structure": None,
                }
                provider = SequenceProvider([envelope])
                frames = stable_frames(color)

                observed = SingleStepGenericSceneObserver(provider).observe(
                    frames=frames,
                    goal_context=goal_context,
                    device_id="device-local-01",
                    available_action_kinds={"back", "home"},
                )

                image_parts = [
                    part for part in provider.messages_seen[0][1]["content"] if part.get("type") == "image_url"
                ]
                self.assertEqual(1, len(image_parts))
                self.assertEqual(
                    _image_data_url(frames[-1].convert("RGB")),
                    image_parts[0]["image_url"]["url"],
                )
                self.assertEqual("calculator", observed.foreground_app_id)
                self.assertEqual("app_home", observed.screen_id)
                prompt = provider.messages_seen[0][1]["content"][0]["text"]
                self.assertNotIn("中央App内容未披露", prompt)
                self.assertNotIn("固定遮罩", prompt)
                self.assertIn(
                    "当前前台App不是当前高层目标的目标App",
                    prompt,
                )
                self.assertIn(
                    "不得仅因目标App入口",
                    prompt,
                )
                self.assertIn(
                    "不在当前App画面而停止",
                    prompt,
                )
                self.assertIn("单个动作只需推进路径", prompt)
                self.assertNotIn('"status":"blocked"', prompt)
                self.assertIn(
                    "back：只返回当前App或当前系统页面的上一层",
                    prompt,
                )
                self.assertIn(
                    "open_recent_apps：打开Android最近任务卡片页",
                    prompt,
                )
                self.assertIn(
                    '当前设备本轮可用动作（唯一运行时动作集合）：["back","home"]',
                    prompt,
                )
                self.assertIn(
                    "若必须退出当前",
                    prompt,
                )
                self.assertIn(
                    "App或返回Launcher/主屏幕才能继续",
                    prompt,
                )
                self.assertIn(
                    "必须选择\n   status=action、action=home",
                    prompt,
                )

    def test_runtime_action_set_is_sent_on_every_observation(self) -> None:
        envelope = {
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {
                "kind": "normalized_1000",
                "width": 1000,
                "height": 1000,
            },
            "scene": scene_payload(),
            "input_structure": None,
        }
        provider = SequenceProvider(
            [
                json.loads(json.dumps(envelope)),
                json.loads(json.dumps(envelope)),
                json.loads(json.dumps(envelope)),
            ]
        )
        observer = SingleStepGenericSceneObserver(provider)
        frames = stable_frames()
        goal_context = {
            "objective": "打开目标应用",
            "execution_class": "navigate",
            "completion_conditions": ["目标应用主页面可见"],
        }

        observer.observe(
            frames=frames,
            goal_context=goal_context,
            device_id="device-local-01",
            available_action_kinds={"home"},
        )
        observer.observe(
            frames=frames,
            goal_context=goal_context,
            device_id="device-local-01",
            available_action_kinds={"back"},
        )
        observer.observe(
            frames=frames,
            goal_context=goal_context,
            device_id="device-local-01",
            available_action_kinds={"back"},
        )

        self.assertEqual(3, provider.calls)
        home_prompt = provider.messages_seen[0][1]["content"][0]["text"]
        back_prompt = provider.messages_seen[1][1]["content"][0]["text"]
        repeated_back_prompt = provider.messages_seen[2][1]["content"][0]["text"]
        self.assertIn(
            '当前设备本轮可用动作（唯一运行时动作集合）：["home"]',
            home_prompt,
        )
        self.assertIn(
            '当前设备本轮可用动作（唯一运行时动作集合）：["back"]',
            back_prompt,
        )
        self.assertIn(
            '当前设备本轮可用动作（唯一运行时动作集合）：["back"]',
            repeated_back_prompt,
        )

    def test_target_only_clear_binds_visible_preedit_to_unique_focused_field(
        self,
    ) -> None:
        scene = scene_payload()
        scene.update(
            {
                "foreground_app_id": "com.example.messaging",
                "screen_id": "conversation",
                "summary": "会话页中唯一输入框已聚焦",
                "elements": [],
            }
        )
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="composer",
                    bounds=[100, 500, 760, 600],
                    text="",
                    placeholder="",
                    visible_editable_cues=["aaazjie"],
                    caret_line_index=0,
                )
            ],
            ime_preedit_regions=[
                {
                    "region_id": "preedit",
                    "bounds": [160, 525, 330, 570],
                    "text": "aaazjie",
                    "confidence": 0.99,
                    "candidates": [
                        {
                            "text": "aaazjie",
                            "bounds": [160, 610, 330, 645],
                            "confidence": 0.99,
                            "fully_visible": True,
                        }
                    ],
                }
            ],
            keyboard={
                "visible": True,
                "bounds": [0, 650, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "case_mode": "lower",
                "mode_switch": None,
                "backspace_key": {
                    "label": "⌫",
                    "bounds": [830, 810, 930, 880],
                    "confidence": 0.99,
                    "fully_visible": True,
                },
            },
        )
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "clear_preedit",
                    "objective": "清空当前唯一聚焦输入框中的预编辑",
                    "constraints": ["不要发送"],
                    "completion_conditions": ["输入框和预编辑都为空"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_field_id": "input_field_clear_target",
                        "active_input_operation": "clear_verified_text",
                        "active_input_target_only": True,
                        "active_input_multiline": False,
                    },
                }
            }
        }

        observed = SingleStepGenericSceneObserver(
            SequenceProvider(
                [
                    {
                        "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                        "coordinate_space": {
                            "kind": "normalized_1000",
                            "width": 1000,
                            "height": 1000,
                        },
                        "scene": scene,
                        "input_structure": audit,
                    }
                ]
            )
        ).observe(
            frames=stable_frames(),
            goal_context=context,
            device_id="device-local-01",
        )

        field = unique_goal_element(observed)
        self.assertIsNotNone(field)
        self.assertEqual(
            "input_field_clear_target",
            field.states["input_field_id"],
        )
        self.assertEqual("aaazjie", field.states["ime_preedit_text"])
        self.assertEqual("", field.states["value"])

    def test_target_only_clear_rejects_unbound_preedit_and_backspace(self) -> None:
        scene = scene_payload()
        scene["elements"] = []
        audit = input_audit_payload(
            ime_preedit_regions=[
                {
                    "region_id": "preedit",
                    "bounds": [160, 525, 330, 570],
                    "text": "aaazjie",
                    "confidence": 0.99,
                    "candidates": [],
                }
            ],
            keyboard={
                "visible": True,
                "bounds": [0, 650, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "case_mode": "lower",
                "mode_switch": None,
                "backspace_key": {
                    "label": "⌫",
                    "bounds": [830, 810, 930, 880],
                    "confidence": 0.99,
                    "fully_visible": True,
                },
            },
        )
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "clear_preedit",
                    "objective": "清空当前唯一聚焦输入框中的预编辑",
                    "constraints": ["不要发送"],
                    "completion_conditions": ["输入框和预编辑都为空"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_field_id": "input_field_clear_target",
                        "active_input_operation": "clear_verified_text",
                        "active_input_target_only": True,
                        "active_input_multiline": False,
                    },
                }
            }
        }

        observed = SingleStepGenericSceneObserver(SequenceProvider([{
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "normalized_1000", "width": 1000, "height": 1000},
            "scene": scene, "input_structure": audit,
        }])).observe(frames=stable_frames(), goal_context=context, device_id="device-local-01")

        self.assertIsNone(unique_goal_element(observed))

    def test_single_step_observer_uses_one_request_for_scene_and_input(self) -> None:
        scene = scene_payload()
        scene.update(
            {
                "foreground_app_id": "wechat",
                "screen_id": "chat",
                "summary": "聊天页输入框可见",
                "elements": [],
            }
        )
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="message",
                    bounds=[100, 720, 900, 820],
                    text="",
                    placeholder="消息",
                    field_labels=["消息"],
                    visible_editable_cues=["消息输入框完整边框", "插入光标"],
                    caret_line_index=0,
                )
            ]
        )
        provider = SequenceProvider(
            [
                {
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    "coordinate_space": {
                        "kind": "normalized_1000",
                        "width": 1000,
                        "height": 1000,
                    },
                    "scene": scene,
                    "input_structure": audit,
                }
            ]
        )
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_message",
                    "objective": "在消息输入框输入abc",
                    "constraints": [],
                    "completion_conditions": ["消息输入框内容为abc"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "abc",
                        "active_input_field_id": "message_field",
                        "active_input_field_label": "消息",
                        "active_input_multiline": False,
                    },
                }
            }
        }

        observed = SingleStepGenericSceneObserver(provider).observe(
            frames=stable_frames(),
            goal_context=context,
            device_id="device-local-01",
        )

        self.assertEqual(provider.calls, 1)
        self.assertEqual(provider.call_options["max_attempts"], 1)
        prompt = json.dumps(provider.messages_seen[0], ensure_ascii=False)
        self.assertIn("A blank input is valid", prompt)
        self.assertIn("does not need visible text, placeholder, caret", prompt)
        self.assertIn("Companion IME input/clear and a focus tap do not require them", prompt)
        self.assertIn("scene中的role=input只是可选页面上下文", prompt)
        self.assertIn('element_id=\\"local_audited_input_1\\"', prompt)
        self.assertEqual(
            unique_goal_element(observed).element_id,
            "local_audited_input_1",
        )

    def test_input_target_miss_does_not_create_an_execution_target(self):
        wrong_scene = scene_payload()
        wrong_scene.update(
            {
                "foreground_app_id": "com.example.messaging",
                "screen_id": "wrong_named_conversation",
                "summary": "进入了相邻会话，当前没有消息输入框",
                "elements": [],
            }
        )
        envelope = {
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {
                "kind": "normalized_1000",
                "width": 1000,
                "height": 1000,
            },
            "scene": wrong_scene,
            "input_structure": input_audit_payload(),
        }
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_message",
                    "objective": "在消息输入框输入abc",
                    "constraints": [],
                    "completion_conditions": ["消息输入框内容为abc"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "abc",
                        "active_input_field_id": "message_field",
                        "active_input_field_label": "消息",
                        "active_input_multiline": False,
                        "observation_phase": "untrusted-successor-preview",
                    },
                }
            }
        }

        provider = SequenceProvider([envelope])
        observed = SingleStepGenericSceneObserver(provider).observe(
            frames=stable_frames(), goal_context=context, device_id="device-local-01")

        self.assertIsNone(unique_goal_element(observed))
        self.assertEqual(1, provider.calls)

    def test_unique_compact_input_miss_is_preserved_only_for_focus(self):
        compact_scene = scene_payload()
        compact_scene.update(
            {
                "foreground_app_id": "com.example.messaging",
                "screen_id": "named_conversation",
                "summary": "指定会话页底部有一个空白编辑面",
                "elements": [
                    {
                        "element_id": "coarse-input",
                        "role": "input",
                        "bounds": [120, 910, 780, 960],
                        "states": {
                            "goal_relevant": False,
                            "fully_visible": True,
                        },
                    }
                ],
            }
        )
        envelope = {
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {
                "kind": "normalized_1000",
                "width": 1000,
                "height": 1000,
            },
            "scene": compact_scene,
            "input_structure": input_audit_payload(),
        }
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_message",
                    "objective": "在消息输入框输入abc",
                    "constraints": [],
                    "completion_conditions": ["消息输入框内容为abc"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "abc",
                        "active_input_field_id": "message_field",
                        "active_input_field_label": "消息",
                        "active_input_multiline": False,
                    },
                }
            }
        }

        for with_spoofed_phase in (False, True):
            with self.subTest(with_spoofed_phase=with_spoofed_phase):
                current_context = json.loads(json.dumps(context, ensure_ascii=False))
                if with_spoofed_phase:
                    current_context["entities"]["active_subgoal_visual_context"]["goal_entities"][
                        "observation_phase"
                    ] = "untrusted-successor-preview"
                observed = SingleStepGenericSceneObserver(SequenceProvider([envelope])).observe(
                    frames=stable_frames(),
                    goal_context=current_context,
                    device_id="device-local-01",
                )

                target = unique_goal_element(observed)
                self.assertIsNotNone(target)
                self.assertEqual("coarse-input", target.element_id)
                self.assertEqual("input", target.role)
                self.assertEqual(
                    {
                        "goal_relevant": True,
                        "fully_visible": True,
                        "focus_only_input_surface": True,
                    },
                    target.states,
                )
                self.assertNotIn("value", target.states)
                self.assertNotIn("input_field_id", target.states)
                self.assertFalse(target.element_id.startswith("local_audited_"))

    def test_ambiguous_or_unbound_focus_surfaces_do_not_create_local_targets_or_rewrite_qwen_facts(self):
        base_input = {
            "element_id": "coarse-input-1",
            "role": "input",
            "meaning": "form_text_field",
            "label": "备注",
            "bounds": [100, 300, 900, 390],
            "confidence": 0.99,
            "states": {"goal_relevant": True, "fully_visible": True},
            "evidence": ["表单中完整可见的备注编辑区域"],
        }
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_notes",
                    "objective": "在备注字段输入release",
                    "constraints": [],
                    "completion_conditions": ["备注字段为release"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "release",
                        "active_input_field_id": "notes_field",
                        "active_input_field_label": "备注",
                        "active_input_multiline": False,
                    },
                }
            }
        }
        cases = (
            (
                "two-inputs",
                [
                    base_input,
                    {
                        **base_input,
                        "element_id": "coarse-input-2",
                        "label": "正文",
                        "bounds": [100, 430, 900, 520],
                        "evidence": ["表单中另一个完整可见的正文编辑区域"],
                    },
                ],
                input_audit_payload(),
            ),
            (
                "keyboard-visible",
                [base_input],
                input_audit_payload(
                    keyboard={
                        "visible": True,
                        "bounds": [50, 600, 950, 990],
                        "layout": "numeric",
                        "input_mode": "unknown",
                        "mode_switch": None,
                    }
                ),
            ),
        )
        for name, elements, audit in cases:
            with self.subTest(name=name):
                candidate_scene = scene_payload()
                candidate_scene.update(
                    {
                        "foreground_app_id": "com.example.form",
                        "screen_id": "edit_form",
                        "elements": elements,
                    }
                )
                envelope = {
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    "coordinate_space": {
                        "kind": "normalized_1000",
                        "width": 1000,
                        "height": 1000,
                    },
                    "scene": candidate_scene,
                    "input_structure": audit,
                }
                observed = SingleStepGenericSceneObserver(SequenceProvider([envelope])).observe(
                    frames=stable_frames(), goal_context=context, device_id="device-local-01")
                self.assertFalse(any(item.element_id.startswith("local_audited_") for item in observed.elements))
                for element_id in {item["element_id"] for item in elements}:
                    self.assertTrue(observed.get_element(element_id).states["goal_relevant"])
                if name == "two-inputs":
                    self.assertIsNone(unique_goal_element(observed))
                else:
                    self.assertEqual("coarse-input-1", unique_goal_element(observed).element_id)

    def test_single_step_observer_uses_only_input_structure_for_blank_value(
        self,
    ) -> None:
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_message",
                    "objective": "在输入框中输入消息",
                    "constraints": [],
                    "completion_conditions": ["输入框显示指定消息"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "aaazjie？你好",
                        "active_input_field_id": "message_field",
                        "active_input_multiline": False,
                    },
                }
            }
        }
        cases = (
            (
                "harmless-extra",
                "com.example.messaging",
                "named_conversation",
                {"bounds": [120, 910, 780, 960], "text": "", "visual_note": "optional metadata"},
            ),
            (
                "low-confidence",
                "com.example.notes",
                "edit_note",
                {"bounds": [100, 300, 900, 390], "confidence": 0.01},
            ),
            (
                "no-fully-visible",
                "com.example.forms",
                "edit_form",
                {"bounds": [80, 420, 920, 510]},
            ),
        )
        for name, app_id, screen_id, audit_item in cases:
            with self.subTest(name=name):
                scene = scene_payload()
                scene.update(
                    {
                        "foreground_app_id": app_id,
                        "screen_id": screen_id,
                        "summary": "当前页面有一个空输入框",
                        "elements": [],
                    }
                )
                focused_audit_item = dict(audit_item)
                focused_audit_item["caret_line_index"] = 0
                audit = input_audit_payload(application_inputs=[focused_audit_item])
                provider = SequenceProvider(
                    [
                        {
                            "protocol_version": (SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION),
                            "coordinate_space": {
                                "kind": "normalized_1000",
                                "width": 1000,
                                "height": 1000,
                            },
                            "scene": scene,
                            "input_structure": audit,
                            "decision": {"status": "action", "action": "input_verified_text",
                                "element_id": "local_audited_input_1"},
                        }
                    ]
                )

                observer = SingleStepGenericSceneObserver(provider)
                observed, decision = observer.observe_with_decision(
                    frames=stable_frames(),
                    goal_context=context,
                    device_id="device-local-01",
                )

                self.assertEqual(provider.calls, 1)
                field = unique_goal_element(observed)
                self.assertEqual("local_audited_input_1", field.element_id)
                self.assertEqual("", field.states["value"])
                self.assertNotIn("same_frame_input_surface_evidence", field.states)
                self.assertEqual("input_verified_text", decision["action"])
                self.assertEqual(1.0, decision["confidence"])
                self.assertTrue(decision["reason"])

    def test_blank_audited_input_focus_follows_affirmative_caret_cues_without_keyboard(
        self,
    ) -> None:
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_text",
                    "objective": "在当前唯一输入框中输入文字",
                    "constraints": [],
                    "completion_conditions": ["输入框显示指定文字"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "sample text",
                        "active_input_field_id": "current_field",
                        "active_input_multiline": False,
                    },
                }
            }
        }
        cases = (
            ("cursor", ["cursor"], None, True),
            ("caret", ["caret"], None, True),
            ("chinese-cursor", ["光标"], None, True),
            ("insertion-mark", ["插入符"], None, True),
            ("insertion-mark-in-field", ["插入符位于空输入框内"], None, True),
            ("caret-line-zero", [], 0, True),
            ("no-cue", [], None, False),
            ("border", ["complete input border"], None, False),
            ("outline", ["input outline"], None, False),
            ("negated-cursor", ["no cursor"], None, False),
            ("hidden-caret", ["caret hidden"], None, False),
            ("invisible-chinese-cursor", ["光标不可见"], None, False),
        )
        for index, (name, cues, caret_line_index, expected_focused) in enumerate(cases, start=1):
            with self.subTest(name=name):
                scene = scene_payload()
                scene.update(
                    {
                        "foreground_app_id": f"com.example.editor{index}",
                        "screen_id": "edit_text",
                        "summary": "当前页面有一个空输入框",
                        "elements": [],
                    }
                )
                audit = input_audit_payload(
                    application_inputs=[
                        audited_application_input(
                            structure_id="current-input",
                            bounds=[100, 720, 900, 820],
                            text="",
                            visible_editable_cues=cues,
                            caret_line_index=caret_line_index,
                        )
                    ],
                    keyboard={
                        "visible": False,
                        "bounds": None,
                        "layout": "unknown",
                        "input_mode": "unknown",
                        "mode_switch": None,
                    },
                )

                observed = SingleStepGenericSceneObserver(SequenceProvider([{
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    "coordinate_space": {
                        "kind": "normalized_1000",
                        "width": 1000,
                        "height": 1000,
                    },
                    "scene": scene,
                    "input_structure": audit,
                }])).observe(
                    frames=stable_frames(),
                    goal_context=context,
                    device_id="device-local-01",
                )

                field = observed.get_element("local_audited_input_1")
                self.assertEqual(expected_focused, field.states.get("focused") is True)
                self.assertFalse(field.states["soft_keyboard_visible"])

    def test_single_step_prompt_requires_focus_before_each_typed_operation(self) -> None:
        for operation in ("input_verified_text", "clear_verified_text", "press_enter"):
            with self.subTest(operation=operation):
                scene = scene_payload()
                scene.update({"foreground_app_id": "com.example.notes", "screen_id": "editor",
                    "summary": "当前页面有一个空白编辑框", "elements": []})
                audit = input_audit_payload(application_inputs=[audited_application_input(
                    structure_id="note-body", bounds=[100, 700, 900, 820], text="",
                    visible_editable_cues=[])])
                provider = SequenceProvider([{
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    "coordinate_space": {"kind": "normalized_1000", "width": 1000, "height": 1000},
                    "scene": scene, "input_structure": audit,
                    "decision": {"status": "action", "action": "tap_semantic",
                        "element_id": "local_audited_input_1"},
                }])
                context = {"entities": {"active_subgoal_visual_context": {
                    "subgoal_id": "edit_note", "objective": "修改指定文字", "constraints": [],
                    "completion_conditions": ["编辑框达到目标状态"], "execution_class": "navigate",
                    "goal_entities": {"active_input_transaction_text": "sample text",
                        "active_input_field_id": "note_body", "active_input_operation": operation,
                        "active_input_multiline": operation == "press_enter"}}}}

                observed, decision = SingleStepGenericSceneObserver(provider).observe_with_decision(
                    frames=stable_frames(), goal_context=context, device_id="device-local-01")

                field = observed.get_element("local_audited_input_1")
                self.assertIsNot(field.states.get("focused"), True)
                self.assertEqual("note_body", field.states["input_field_id"])
                self.assertEqual("tap_semantic", decision["action"])
                prompt = json.dumps(provider.messages_seen[0], ensure_ascii=False)
                self.assertIn("必须先选择tap_semantic", prompt)
                self.assertIn("执行后等待下一张新截图确认聚焦", prompt)
                self.assertIn("states.input_element_id绑定该输入框", prompt)
                self.assertIn("caret_line_index为合法行号", prompt)
                self.assertIn("不得把这两个物理步骤合并成一个动作", prompt)

    def test_same_response_selected_scene_input_keeps_b425_element_identity(self) -> None:
        scene = scene_payload()
        scene.update({
            "foreground_app_id": "com.tencent.mm",
            "screen_id": "chat_file_transfer_assistant",
            "summary": "文件传输助手聊天界面，底部空白输入框可见",
            "elements": [{
                "element_id": "e2",
                "role": "input",
                "meaning": "message_input_field",
                "label": "",
                "bounds": [120, 1160, 780, 1230],
                "confidence": 1.0,
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": ["底部工具栏中央空白长条区域"],
            }],
        })
        context = {"entities": {"active_subgoal_visual_context": {
            "subgoal_id": "input_text",
            "objective": "在消息输入框输入 aaazjie？你好",
            "constraints": [],
            "completion_conditions": ["输入框内容为 aaazjie？你好"],
            "execution_class": "navigate",
            "goal_entities": {
                "active_input_transaction_text": "aaazjie？你好",
                "active_input_field_id": "message_field",
                "active_input_multiline": False,
            },
        }}}
        envelope = {
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "axis_grid", "width": 1000, "height": 1280},
            "scene": scene,
            "input_structure": input_audit_payload(
                application_inputs=[{"bounds": [120, 1160, 780, 1230], "text": ""}],
            ),
            "decision": {"status": "action", "action": "tap_semantic", "element_id": "e2"},
        }
        observer = SingleStepGenericSceneObserver(SequenceProvider([envelope]))

        observed, model_decision = observer.observe_with_decision(
            frames=[Image.new("RGB", (720, 1280), (30, 40, 50)) for _ in range(4)],
            goal_context=context,
            device_id="device-local-01",
        )

        field = observed.get_element("e2")
        self.assertIsNotNone(field)
        self.assertEqual("", field.states["value"])
        self.assertFalse(any(item.element_id == "local_audited_input_1" for item in observed.elements))
        self.assertEqual("e2", model_decision["element_id"])

    def test_input_audit_uses_current_text_and_ignores_qwen_input_goal_marker(self) -> None:
        context = {"entities": {"active_subgoal_visual_context": {
            "subgoal_id": "input_text",
            "objective": "在当前输入框继续输入",
            "constraints": [],
            "completion_conditions": ["输入框显示目标文字"],
            "execution_class": "navigate",
            "goal_entities": {
                "active_input_transaction_text": "fresh-value",
                "active_input_field_id": "message_field",
                "active_input_multiline": False,
            },
        }}}
        for current_text, other_marker in (("fresh-value", False), ("first\nsecond", True)):
            with self.subTest(current_text=current_text):
                current_context = json.loads(json.dumps(context, ensure_ascii=False))
                current_context["entities"]["active_subgoal_visual_context"]["goal_entities"][
                    "active_input_transaction_text"] = current_text
                scene = scene_payload()
                scene.update({
                    "foreground_app_id": "com.example.messaging",
                    "screen_id": "conversation",
                    "summary": "当前页面有一个消息输入框和一段页面说明",
                    "elements": [{
                        "element_id": "message-input",
                        "role": "input",
                        "meaning": "application_text_input",
                        "label": "",
                        "bounds": [100, 500, 800, 590],
                        "confidence": 1.0,
                        "states": {"goal_relevant": False, "fully_visible": True,
                            "value": "stale-scene-value"},
                        "evidence": ["当前截图中的完整编辑栏"],
                    }, {
                        "element_id": "page-note",
                        "role": "text",
                        "meaning": "page_note",
                        "label": "页面说明",
                        "bounds": [100, 200, 500, 260],
                        "confidence": 1.0,
                        "states": {"goal_relevant": other_marker, "fully_visible": True},
                        "evidence": ["页面说明逐字可见"],
                    }],
                })
                audit = input_audit_payload(application_inputs=[{
                    "element_id": "message-input",
                    "bounds": [100, 500, 800, 590],
                    "text": current_text,
                    "visible_editable_cues": ["stale-lineage-value", "cursor"],
                }])
                observer = SingleStepGenericSceneObserver(SequenceProvider([{
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    "coordinate_space": {"kind": "normalized_1000", "width": 1000, "height": 1000},
                    "scene": scene,
                    "input_structure": audit,
                    "decision": {"status": "action", "action": "input_verified_text",
                        "element_id": "message-input"},
                }]))

                observed = observer.observe(
                    frames=stable_frames(), goal_context=current_context, device_id="device-local-01")

                field = observed.get_element("message-input")
                self.assertEqual(current_text, field.states["value"])
                self.assertIs(field.states["goal_relevant"], True)
                self.assertIs(other_marker, observed.get_element("page-note").states["goal_relevant"])

    def test_symbol_routing_switches_to_english_before_symbol_layout(self) -> None:
        current = "aaazjie"
        target = current + "？"
        audited = _apply_input_structure_audit(
            keyboard_routing_scene(current, "symbol-mode-before"),
            keyboard_routing_audit(
                value=current,
                layout="qwerty",
                input_mode="chinese_pinyin",
                mode_switch={
                    "label": "中/英",
                    "bounds": [820, 900, 960, 960],
                    "confidence": 1.0,
                    "current_mode": "chinese_pinyin",
                    "target_mode": "direct_latin",
                },
                layout_switches=[
                    {"label": "123", "bounds": [20, 900, 160, 960], "confidence": 1.0,
                        "current_layout": "qwerty", "target_layout": "numeric"},
                    {"label": "！？#", "bounds": [180, 900, 340, 960], "confidence": 1.0,
                        "current_layout": "qwerty", "target_layout": "symbol"},
                ],
            ),
            fingerprint="symbol-mode-before",
            goal_context={"objective": f"让输入框逐字显示 {target}", "entities": {"input_text": target}},
        )
        goal_elements = [item for item in audited.elements if item.states.get("goal_relevant") is True]
        self.assertEqual(["switch_keyboard_input_mode"], [item.meaning for item in goal_elements])
        self.assertEqual("direct_latin", goal_elements[0].states["target_mode"])

    def test_symbol_and_digit_use_distinct_visible_layout_switches(self) -> None:
        current = "aaazjie"
        switches = [
            {"label": "123", "bounds": [20, 900, 160, 960], "confidence": 1.0,
                "current_layout": "qwerty", "target_layout": "numeric"},
            {"label": "！？#", "bounds": [180, 900, 340, 960], "confidence": 1.0,
                "current_layout": "qwerty", "target_layout": "symbol"},
        ]
        for suffix, expected_layout, expected_label in (("？", "symbol", "！？#"), ("1", "numeric", "123")):
            with self.subTest(suffix=suffix):
                target = current + suffix
                audited = _apply_input_structure_audit(
                    keyboard_routing_scene(current, f"layout-{expected_layout}"),
                    keyboard_routing_audit(value=current, layout="qwerty", input_mode="direct_latin",
                        layout_switches=switches),
                    fingerprint=f"layout-{expected_layout}",
                    goal_context={"objective": f"让输入框逐字显示 {target}",
                        "entities": {"input_text": target}},
                )
                layout_element = audited.get_element("local_audited_keyboard_layout_switch_1")
                self.assertEqual(expected_label, layout_element.label)
                self.assertEqual(expected_layout, layout_element.states["target_layout"])

    def test_symbol_routing_never_uses_123_as_an_implicit_hop(self) -> None:
        current = "aaazjie"
        target = current + "？"
        audited = _apply_input_structure_audit(
            keyboard_routing_scene(current, "symbol-no-switch"),
            keyboard_routing_audit(
                value=current,
                layout="qwerty",
                input_mode="direct_latin",
                layout_switches=[{"label": "123", "bounds": [20, 900, 160, 960], "confidence": 1.0,
                    "current_layout": "qwerty", "target_layout": "numeric"}],
            ),
            fingerprint="symbol-no-switch",
            goal_context={"objective": f"让输入框逐字显示 {target}", "entities": {"input_text": target}},
        )
        self.assertFalse(any(item.meaning == "switch_keyboard_layout"
            and item.states.get("goal_relevant") is True for item in audited.elements))

    def test_symbol_layout_exposes_only_the_exact_requested_character(self) -> None:
        current = "aaazjie"
        target = current + "？"
        audited = _apply_input_structure_audit(
            keyboard_routing_scene(current, "symbol-exact-key"),
            keyboard_routing_audit(value=current, layout="symbol", input_mode="direct_latin", literal="？"),
            fingerprint="symbol-exact-key",
            goal_context={"objective": f"让输入框逐字显示 {target}", "entities": {"input_text": target}},
        )
        literal = audited.get_element("local_audited_literal_key_1")
        self.assertEqual("？", literal.label)
        self.assertEqual("？", literal.states["key_value"])
        self.assertEqual(target, literal.states["expected_input_value"])

    def test_cross_app_blank_form_keeps_scene_identity_and_input_structure_value(self) -> None:
        scene = scene_payload()
        scene.update({
            "foreground_app_id": "com.example.forms",
            "screen_id": "new_contact_form",
            "summary": "联系人表单中唯一空白备注输入框可见",
            "elements": [{
                "element_id": "form-note",
                "role": "input",
                "meaning": "note_input_field",
                "label": "备注",
                "bounds": [90, 300, 910, 400],
                "confidence": 0.82,
                "states": {
                    "goal_relevant": True,
                    "fully_visible": True,
                    "value": "scene-transcription-must-not-win",
                },
                "evidence": ["备注标签右侧完整编辑栏"],
            }],
        })
        context = {"entities": {"active_subgoal_visual_context": {
            "subgoal_id": "input_note",
            "objective": "在备注输入框输入 follow-up",
            "constraints": [],
            "completion_conditions": ["备注输入框内容为 follow-up"],
            "execution_class": "navigate",
            "goal_entities": {
                "active_input_transaction_text": "follow-up",
                "active_input_field_id": "note_field",
                "active_input_field_label": "备注",
                "active_input_multiline": False,
            },
        }}}
        provider = SequenceProvider([{
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "normalized_1000", "width": 1000, "height": 1000},
            "scene": scene,
            "input_structure": input_audit_payload(application_inputs=[{
                "element_id": "form-note",
                "bounds": [90, 300, 910, 400],
                "text": "",
            }]),
            "decision": {"status": "action", "action": "tap_semantic", "element_id": "form-note"},
        }])
        observer = SingleStepGenericSceneObserver(provider)

        observed, model_decision = observer.observe_with_decision(
            frames=stable_frames(), goal_context=context, device_id="device-local-01")

        field = observed.get_element("form-note")
        self.assertIsNotNone(field)
        self.assertEqual("", field.states["value"])
        self.assertEqual("note_field", field.states["input_field_id"])
        self.assertFalse(any(item.element_id == "local_audited_input_1" for item in observed.elements))
        self.assertEqual("form-note", model_decision["element_id"])

    def test_blank_input_with_invalid_bounds_is_still_rejected(self) -> None:
        scene = scene_payload()
        scene.update(
            {
                "foreground_app_id": "com.example.forms",
                "screen_id": "edit_form",
                "summary": "当前页面有一个空输入框",
                "elements": [],
            }
        )
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_message",
                    "objective": "在输入框输入abc",
                    "constraints": [],
                    "completion_conditions": ["输入框内容为abc"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "abc",
                        "active_input_field_id": "message_field",
                        "active_input_multiline": False,
                    },
                }
            }
        }
        provider = SequenceProvider(
            [
                {
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    "coordinate_space": {
                        "kind": "normalized_1000",
                        "width": 1000,
                        "height": 1000,
                    },
                    "scene": scene,
                    "input_structure": input_audit_payload(
                        application_inputs=[{"bounds": [100, 300, 100, 390]}]
                    ),
                    "decision": {
                        "status": "action",
                        "action": "input_verified_text",
                        "element_id": "local_audited_input_1",
                    },
                }
            ]
        )

        with self.assertRaisesRegex(VisionAgentError, "bounds"):
            SingleStepGenericSceneObserver(provider).observe(
                frames=stable_frames(),
                goal_context=context,
                device_id="device-local-01",
            )

    def test_optional_invalid_and_surplus_input_facts_do_not_veto_valid_blank_input(self) -> None:
        scene = scene_payload()
        scene.update({
            "foreground_app_id": "com.example.forms",
            "screen_id": "edit_form",
            "summary": "当前页面有一个空输入框",
            "elements": [],
        })
        context = {"entities": {"active_subgoal_visual_context": {
            "subgoal_id": "input_message",
            "objective": "在输入框输入abc",
            "constraints": [],
            "completion_conditions": ["输入框内容为abc"],
            "execution_class": "navigate",
            "goal_entities": {
                "active_input_transaction_text": "abc",
                "active_input_field_id": "message_field",
                "active_input_multiline": False,
            },
        }}}
        invalid_inputs = [{"bounds": [100, 300, 100, 390], "text": "noise"} for _ in range(6)]
        invalid_preedits = [{
            "region_id": f"optional-{index}",
            "bounds": [100, 500, 100, 550],
            "text": "noise",
            "candidates": [{"text": "noise", "bounds": [80, 610, 80, 650]}] * 10,
        } for index in range(6)]
        audit = input_audit_payload(
            application_inputs=[{"bounds": [100, 300, 900, 390], "text": ""}, *invalid_inputs],
            ime_preedit_regions=invalid_preedits,
            keyboard={
                "visible": True,
                "bounds": [0, 650, 0, 1000],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "mode_switch": None,
            },
        )
        provider = SequenceProvider([{
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "normalized_1000", "width": 1000, "height": 1000},
            "scene": scene,
            "input_structure": audit,
            "decision": {"status": "action", "action": "input_verified_text",
                "element_id": "local_audited_input_1"},
        }])

        observed = SingleStepGenericSceneObserver(provider).observe(
            frames=stable_frames(), goal_context=context, device_id="device-local-01")

        field = unique_goal_element(observed)
        self.assertEqual("local_audited_input_1", field.element_id)
        self.assertEqual("", field.states["value"])

    def test_selected_scene_element_with_invalid_bounds_is_rejected(self) -> None:
        scene = scene_payload()
        scene["elements"] = [{
            "element_id": "selected-target",
            "role": "button",
            "meaning": "open_target",
            "label": "打开",
            "bounds": [200, 400, 200, 500],
            "states": {"goal_relevant": True, "visible": True, "enabled": True},
        }]
        provider = SequenceProvider([{
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "normalized_1000", "width": 1000, "height": 1000},
            "scene": scene,
            "decision": {"status": "action", "action": "tap_semantic",
                "element_id": "selected-target"},
        }])

        with self.assertRaisesRegex(VisionAgentError, "已选目标.*bounds"):
            SingleStepGenericSceneObserver(provider).observe(
                frames=stable_frames(), goal_context={"objective": "打开目标"},
                device_id="device-local-01")

    def test_selected_scene_element_id_must_be_unique(self) -> None:
        scene = scene_payload()
        selected = {
            "element_id": "selected-target",
            "role": "button",
            "meaning": "open_target",
            "label": "打开",
            "bounds": [200, 400, 500, 500],
            "states": {"goal_relevant": True, "visible": True, "enabled": True},
        }
        scene["elements"] = [selected, {**selected, "bounds": [550, 400, 850, 500]}]
        provider = SequenceProvider([{
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "normalized_1000", "width": 1000, "height": 1000},
            "scene": scene,
            "decision": {"status": "action", "action": "tap_semantic",
                "element_id": "selected-target"},
        }])

        with self.assertRaisesRegex(VisionAgentError, "element_id不唯一"):
            SingleStepGenericSceneObserver(provider).observe(
                frames=stable_frames(), goal_context={"objective": "打开目标"},
                device_id="device-local-01")

    def test_invalid_unselected_goal_hint_does_not_veto_selected_scene_element(self) -> None:
        scene = scene_payload()
        scene["elements"][0].update(
            element_id="selected-target", meaning="open_target",
            states={"goal_relevant": True, "visible": True, "enabled": True})
        scene["elements"].append({
            "element_id": "bad-optional-hint",
            "role": "button",
            "meaning": "other_target",
            "bounds": [800, 300, 700, 400],
            "states": {"goal_relevant": True},
        })
        provider = SequenceProvider([{
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "normalized_1000", "width": 1000, "height": 1000},
            "scene": scene,
            "decision": {"status": "action", "action": "tap_semantic",
                "element_id": "selected-target"},
        }])

        observed = SingleStepGenericSceneObserver(provider).observe(
            frames=stable_frames(), goal_context={"objective": "打开目标"},
            device_id="device-local-01")

        self.assertEqual(["selected-target"], [item.element_id for item in observed.elements])

    def test_malformed_optional_states_do_not_veto_selected_scene_element(self) -> None:
        scene = scene_payload()
        scene["elements"] = [{
            "element_id": "selected-target",
            "role": "button",
            "meaning": "open_target",
            "label": "打开",
            "bounds": [200, 400, 500, 500],
            "confidence": 0.99,
            "states": {"keyboard_layout": "qwerty"},
            "evidence": ["当前截图中完整可见的打开按钮"],
        }]
        provider = SequenceProvider([{
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "normalized_1000", "width": 1000, "height": 1000},
            "scene": scene,
            "decision": {"status": "action", "action": "tap_semantic",
                "element_id": "selected-target"},
        }])

        observed = SingleStepGenericSceneObserver(provider).observe(
            frames=stable_frames(), goal_context={"objective": "打开目标"},
            device_id="device-local-01")

        selected = observed.get_element("selected-target")
        self.assertEqual("open_target", selected.meaning)
        self.assertEqual({}, selected.states)
        self.assertEqual(1, provider.calls)

    def test_invalid_unselected_element_semantics_are_dropped_without_veto(self) -> None:
        scene = scene_payload()
        scene["elements"][0].update(
            element_id="selected-target", meaning="open_target",
            states={"goal_relevant": True, "visible": True, "enabled": True})
        scene["elements"].append({
            "element_id": "bad-optional-semantic-hint",
            "role": "future_widget",
            "meaning": "unrelated_hint",
            "bounds": [100, 600, 400, 700],
            "confidence": 1.0,
            "states": {"goal_relevant": True},
        })
        provider = SequenceProvider([{
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "normalized_1000", "width": 1000, "height": 1000},
            "scene": scene,
            "decision": {"status": "action", "action": "tap_semantic",
                "element_id": "selected-target"},
        }])

        observed = SingleStepGenericSceneObserver(provider).observe(
            frames=stable_frames(), goal_context={"objective": "打开目标"},
            device_id="device-local-01")

        self.assertEqual(["selected-target"], [item.element_id for item in observed.elements])
        self.assertEqual(1, provider.calls)

    def test_action_referenced_element_with_invalid_semantics_is_rejected(self) -> None:
        scene = scene_payload()
        scene["elements"] = [{
            "element_id": "selected-target",
            "role": "future_widget",
            "meaning": "open_target",
            "bounds": [100, 300, 500, 400],
            "confidence": 1.0,
            "states": {"goal_relevant": True},
        }]
        provider = SequenceProvider([{
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "normalized_1000", "width": 1000, "height": 1000},
            "scene": scene,
            "decision": {"status": "action", "action": "tap_semantic",
                "element_id": "selected-target"},
        }])

        with self.assertRaisesRegex(VisionAgentError, "不支持的元素角色"):
            SingleStepGenericSceneObserver(provider).observe(
                frames=stable_frames(), goal_context={"objective": "打开目标"},
                device_id="device-local-01")

        self.assertEqual(1, provider.calls)

    def test_finish_referenced_element_with_invalid_semantics_is_rejected(self) -> None:
        scene = scene_payload()
        scene["elements"] = [{
            "element_id": "bad-proof",
            "role": "text",
            "meaning": "",
            "label": "已完成",
            "bounds": [100, 300, 500, 400],
            "confidence": 1.0,
            "states": {"goal_relevant": True},
        }]
        provider = SequenceProvider([{
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "normalized_1000", "width": 1000, "height": 1000},
            "scene": scene,
            "decision": {"status": "finish", "evidence_refs": ["element:bad-proof"]},
        }])

        with self.assertRaisesRegex(VisionAgentError, "缺少语义 meaning"):
            SingleStepGenericSceneObserver(provider).observe(
                frames=stable_frames(), goal_context={"objective": "确认目标完成"},
                device_id="device-local-01")

        self.assertEqual(1, provider.calls)

    def test_local_obstruction_is_diagnostic_and_does_not_rewrite_qwen_input_fact(self) -> None:
        scene = scene_payload()
        scene["elements"] = [{
            "element_id": "search-field",
            "role": "input",
            "meaning": "search_input",
            "label": "搜索",
            "bounds": [100, 50, 700, 150],
            "confidence": 1.0,
            "states": {"goal_relevant": True, "fully_visible": True, "focused": True},
            "evidence": ["当前截图显示完整搜索框"],
        }]
        provider = SequenceProvider([{
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "normalized_1000", "width": 1000, "height": 1000},
            "scene": scene,
            "decision": {"status": "action", "action": "tap_semantic",
                "element_id": "search-field"},
        }])
        obstruction = VisualObstruction(kind="top_edge_opaque_band", bounds=(100, 0, 700, 100),
            reason="本地检测到顶边遮挡")

        with patch("agent.infrastructure.generic_scene_observer.consensus_top_edge_obstructions",
            return_value=(obstruction,)):
            observer = SingleStepGenericSceneObserver(provider)
            observed = observer.observe(
                frames=stable_frames(), goal_context={"objective": "聚焦搜索框"},
                device_id="device-local-01")

        field = observed.get_element("search-field")
        self.assertTrue(field.states["fully_visible"])
        self.assertTrue(field.states["goal_relevant"])
        self.assertTrue(field.states["focused"])
        self.assertEqual([obstruction.to_dict()], observer.last_diagnostics["local_visual_obstructions"])

    def test_single_step_observer_keeps_input_when_optional_preedit_bounds_are_broad(
        self,
    ) -> None:
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_message",
                    "objective": "在输入框中输入指定文字",
                    "constraints": [],
                    "completion_conditions": ["输入框显示指定文字"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "aaazjie？你好",
                        "active_input_field_id": "message_field",
                        "active_input_multiline": False,
                    },
                }
            }
        }
        for app_id, screen_id in (
            ("com.example.messaging", "named_conversation"),
            ("com.example.notes", "edit_note"),
        ):
            with self.subTest(app_id=app_id, screen_id=screen_id):
                input_bounds = [130, 500, 720, 590]
                scene = scene_payload()
                scene.update(
                    {
                        "foreground_app_id": app_id,
                        "screen_id": screen_id,
                        "summary": "当前页面的唯一输入框内有带下划线的拉丁预编辑",
                        "elements": [
                            {
                                "element_id": "e1",
                                "role": "input",
                                "meaning": "application_text_input",
                                "label": "",
                                "bounds": input_bounds,
                                "confidence": 1.0,
                                "states": {
                                    "goal_relevant": True,
                                    "fully_visible": True,
                                },
                                "evidence": ["唯一完整输入表面内可见下划线 aaazjie"],
                            }
                        ],
                    }
                )
                audit = input_audit_payload(
                    application_inputs=[
                        audited_application_input(
                            structure_id="message",
                            bounds=input_bounds,
                            text="",
                            placeholder="",
                            visible_editable_cues=["aaazjie"],
                            caret_line_index=0,
                        )
                    ],
                    ime_preedit_regions=[
                        {
                            "region_id": "preedit",
                            # The optional preedit geometry is deliberately as
                            # broad as the whole input surface.  Its own
                            # inaccuracy must not erase the independently
                            # established application input or exact candidate.
                            "bounds": input_bounds,
                            "text": "aaazjie",
                            "confidence": 0.99,
                            "candidates": [
                                {
                                    "text": "aaazjie",
                                    "bounds": [80, 610, 300, 645],
                                    "confidence": 0.99,
                                    "fully_visible": True,
                                }
                            ],
                        }
                    ],
                    keyboard={
                        "visible": True,
                        "bounds": [0, 650, 1000, 1000],
                        "layout": "qwerty",
                        "input_mode": "direct_latin",
                        "case_mode": "lower",
                        "mode_switch": None,
                    },
                )
                observed = SingleStepGenericSceneObserver(
                    SequenceProvider(
                        [
                            {
                                "protocol_version": (SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION),
                                "coordinate_space": {
                                    "kind": "normalized_1000",
                                    "width": 1000,
                                    "height": 1000,
                                },
                                "scene": scene,
                                "input_structure": audit,
                            }
                        ]
                    )
                ).observe(
                    frames=stable_frames(),
                    goal_context=context,
                    device_id="device-local-01",
                )

                field = observed.get_element("e1")
                candidates = [element for element in observed.elements if element.meaning == "ime_exact_candidate"]
                self.assertIsNotNone(field)
                self.assertEqual("", field.states["value"])
                self.assertEqual("aaazjie", field.states["ime_preedit_text"])
                self.assertEqual(1, len(candidates))
                self.assertEqual("e1", candidates[0].states["input_element_id"])
                self.assertEqual("aaazjie", candidates[0].label)
                self.assertEqual((0.08, 0.61, 0.3, 0.645), candidates[0].bounds)

    def test_single_step_observer_rejects_retired_image_grid_contract(
        self,
    ) -> None:
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_message",
                    "objective": "在输入框中输入消息",
                    "constraints": [],
                    "completion_conditions": ["输入框显示指定消息"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "aaazjie？你好",
                        "active_input_field_id": "message_field",
                        "active_input_multiline": False,
                    },
                }
            }
        }
        scene, audit = audited_text_input_scene([120, 870, 780, 915])
        retired = {
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {
                "kind": "image_grid",
                "width": 540,
                "height": 960,
            },
            "scene": scene,
            "input_structure": audit,
        }
        provider = SequenceProvider([json.dumps(retired, ensure_ascii=False)])

        with self.assertRaisesRegex(VisionAgentError, "coordinate_space"):
            SingleStepGenericSceneObserver(provider).observe(
                frames=stable_frames(),
                goal_context=context,
                device_id="device-local-01",
            )

        self.assertEqual(1, provider.calls)

    def test_single_step_observer_normalizes_request_bound_y_axis_grid(
        self,
    ) -> None:
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_message",
                    "objective": "在输入框中输入消息",
                    "constraints": [],
                    "completion_conditions": ["输入框显示指定消息"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "aaazjie？你好",
                        "active_input_field_id": "message_field",
                        "active_input_multiline": False,
                    },
                }
            }
        }
        cases = (
            ((810, 1440), (720, 1280), [120, 1160, 780, 1220]),
            ((540, 960), (540, 960), [120, 870, 780, 915]),
        )
        canonical_bounds = []
        for frame_size, request_size, raw_bounds in cases:
            with self.subTest(request_size=request_size):
                scene, audit = audited_text_input_scene(raw_bounds)
                provider = SequenceProvider(
                    [
                        {
                            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                            "coordinate_space": {
                                "kind": "axis_grid",
                                "width": 1000,
                                "height": request_size[1],
                            },
                            "scene": scene,
                            "input_structure": audit,
                        }
                    ]
                )
                observer = SingleStepGenericSceneObserver(provider)

                observed = observer.observe(
                    frames=[Image.new("RGB", frame_size, (30, 40, 50)) for _ in range(4)],
                    goal_context=context,
                    device_id="device-local-01",
                )

                field = unique_goal_element(observed)
                canonical_bounds.append(tuple(round(value, 3) for value in field.bounds))
                normalization = observer.last_diagnostics["coordinate_normalization"]
                self.assertEqual("axis_grid", normalization["wire_kind"])
                self.assertEqual([1000, request_size[1]], normalization["wire_extent"])
                self.assertEqual(list(request_size), normalization["request_image_size"])
                self.assertTrue(normalization["applied"])
                self.assertEqual(1, provider.calls)
                prompt = provider.messages_seen[0][1]["content"][0]["text"]
                self.assertIn(
                    f'"coordinate_space":{{"kind":"axis_grid","width":1000,' f'"height":{request_size[1]}}}',
                    prompt,
                )
                self.assertNotIn(
                    '"coordinate_space":{"kind":"normalized_1000",' '"width":1000,"height":1000}',
                    prompt,
                )

        self.assertEqual((0.12, 0.906, 0.78, 0.953), canonical_bounds[0])
        self.assertEqual(canonical_bounds[0], canonical_bounds[1])

    def test_single_step_observer_rejects_retired_normalized_y_declaration(
        self,
    ) -> None:
        raw_bounds = [120, 870, 780, 915]
        scene, audit = audited_text_input_scene(raw_bounds)
        retired = {
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {
                "kind": "normalized_1000",
                "width": 1000,
                "height": 1000,
            },
            "scene": scene,
            "input_structure": audit,
        }
        provider = SequenceProvider([json.dumps(retired, ensure_ascii=False)])
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_message",
                    "objective": "在输入框中输入消息",
                    "constraints": [],
                    "completion_conditions": ["输入框显示指定消息"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "aaazjie？你好",
                        "active_input_field_id": "message_field",
                        "active_input_multiline": False,
                    },
                }
            }
        }

        with self.assertRaisesRegex(VisionAgentError, "coordinate_space"):
            SingleStepGenericSceneObserver(provider).observe(
                frames=stable_frames(),
                goal_context=context,
                device_id="device-local-01",
            )

        self.assertEqual(1, provider.calls)

    def test_single_step_observer_rejects_unprovable_wire_coordinate_spaces(
        self,
    ) -> None:
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_message",
                    "objective": "在输入框输入abc",
                    "constraints": [],
                    "completion_conditions": ["输入框内容为abc"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "abc",
                        "active_input_field_id": "message_field",
                        "active_input_multiline": False,
                    },
                }
            }
        }
        base_scene, audit = audited_text_input_scene([120, 870, 780, 915])
        cases = (
            (
                "retired_image_grid",
                {"kind": "image_grid", "width": 540, "height": 960},
                [120, 870, 780, 915],
                "coordinate_space",
            ),
            (
                "retired_normalized_grid",
                {"kind": "normalized_1000", "width": 1000, "height": 1000},
                [120, 870, 780, 915],
                "coordinate_space",
            ),
            (
                "wrong_axis_height",
                {"kind": "axis_grid", "width": 1000, "height": 1280},
                [120, 870, 780, 915],
                "coordinate_space",
            ),
        )
        for name, coordinate_space, bounds, error in cases:
            with self.subTest(name=name):
                scene = json.loads(json.dumps(base_scene, ensure_ascii=False))
                current_audit = json.loads(json.dumps(audit, ensure_ascii=False))
                scene["elements"][0]["bounds"] = list(bounds)
                current_audit["application_inputs"][0]["bounds"] = list(bounds)
                raw = {
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    "coordinate_space": coordinate_space,
                    "scene": scene,
                    "input_structure": current_audit,
                }
                provider = SequenceProvider([json.dumps(raw, ensure_ascii=False)])
                with self.assertRaisesRegex(VisionAgentError, error):
                    SingleStepGenericSceneObserver(provider).observe(
                        frames=stable_frames(),
                        goal_context=context,
                        device_id="device-local-01",
                    )
                self.assertEqual(1, provider.calls)

        provider = SequenceProvider(
            [
                json.dumps(
                    {
                        "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                        "scene": base_scene,
                        "input_structure": audit,
                    },
                    ensure_ascii=False,
                )
            ]
        )
        with self.assertRaisesRegex(VisionAgentError, "coordinate_space"):
            SingleStepGenericSceneObserver(provider).observe(
                frames=stable_frames(),
                goal_context=context,
                device_id="device-local-01",
            )
        self.assertEqual(1, provider.calls)

    def test_single_step_observer_fills_the_only_missing_nested_audit_version(self) -> None:
        scene = scene_payload()
        scene.update(
            {
                "foreground_app_id": "sample_chat",
                "screen_id": "conversation",
                "summary": "消息输入框可见",
                "elements": [],
            }
        )
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="message",
                    bounds=[100, 720, 900, 820],
                    text="",
                    placeholder="消息",
                    field_labels=["消息"],
                    visible_editable_cues=["消息输入框完整边框", "插入光标"],
                    caret_line_index=0,
                )
            ]
        )
        audit.pop("protocol_version")
        provider = SequenceProvider(
            [
                {
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    "coordinate_space": {
                        "kind": "normalized_1000",
                        "width": 1000,
                        "height": 1000,
                    },
                    "scene": scene,
                    "input_structure": audit,
                }
            ]
        )
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_message",
                    "objective": "在消息输入框输入abc",
                    "constraints": [],
                    "completion_conditions": ["消息输入框内容为abc"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "abc",
                        "active_input_field_id": "message_field",
                        "active_input_field_label": "消息",
                        "active_input_multiline": False,
                    },
                }
            }
        }

        observed = SingleStepGenericSceneObserver(provider).observe(
            frames=stable_frames(), goal_context=context, device_id="device-local-01")

        self.assertEqual(provider.calls, 1)
        self.assertEqual("local_audited_input_1", unique_goal_element(observed).element_id)

    def test_single_step_observer_ignores_harmless_nested_audit_metadata(self) -> None:
        scene = scene_payload()
        audit = input_audit_payload(application_inputs=[{"bounds": [100, 720, 900, 820], "text": ""}])
        audit.pop("protocol_version")
        audit["unexpected"] = True
        provider = SequenceProvider(
            [
                {
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    "coordinate_space": {
                        "kind": "normalized_1000",
                        "width": 1000,
                        "height": 1000,
                    },
                    "scene": scene,
                    "input_structure": audit,
                }
            ]
        )
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_message",
                    "objective": "在消息输入框输入abc",
                    "constraints": [],
                    "completion_conditions": ["消息输入框内容为abc"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "abc",
                        "active_input_field_id": "message_field",
                        "active_input_field_label": "消息",
                        "active_input_multiline": False,
                    },
                }
            }
        }

        observed = SingleStepGenericSceneObserver(provider).observe(
            frames=stable_frames(), goal_context=context, device_id="device-local-01")

        self.assertEqual(provider.calls, 1)
        self.assertEqual("local_audited_input_1", unique_goal_element(observed).element_id)

    def test_single_step_observer_rejects_explicit_conflicting_nested_audit_version(self) -> None:
        scene = scene_payload()
        audit = input_audit_payload(application_inputs=[])
        audit["protocol_version"] = "conflicting-version"
        provider = SequenceProvider([{
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "normalized_1000", "width": 1000, "height": 1000},
            "scene": scene, "input_structure": audit,
        }])
        context = {"entities": {"active_subgoal_visual_context": {
            "subgoal_id": "input_message", "objective": "在消息输入框输入abc", "constraints": [],
            "completion_conditions": ["消息输入框内容为abc"], "execution_class": "navigate",
            "goal_entities": {"active_input_transaction_text": "abc",
                "active_input_field_id": "message_field", "active_input_multiline": False},
        }}}

        with self.assertRaisesRegex(VisionAgentError, "协议版本不匹配"):
            SingleStepGenericSceneObserver(provider).observe(
                frames=stable_frames(), goal_context=context, device_id="device-local-01")
        self.assertEqual(1, provider.calls)

    def test_single_step_observer_does_not_retry_malformed_response(self) -> None:
        provider = SequenceProvider(["{not-json"])
        observer = SingleStepGenericSceneObserver(provider)

        with self.assertRaises(VisionAgentError):
            observer.observe(frames=stable_frames())

        self.assertEqual(provider.calls, 1)
        self.assertEqual(observer.last_diagnostics["model_calls"], 1)
        self.assertFalse(observer.last_diagnostics["remote_retry_used"])

    def test_single_step_observer_rejects_retired_flat_scene(self) -> None:
        provider = SequenceProvider([json.dumps(scene_payload(), ensure_ascii=False)])
        observer = SingleStepGenericSceneObserver(provider)

        with self.assertRaisesRegex(VisionAgentError, "coordinate_space"):
            observer.observe(frames=stable_frames())

        self.assertEqual(provider.calls, 1)
        self.assertEqual(observer.last_diagnostics["model_calls"], 1)

    def test_single_step_non_input_ignores_unselected_overflow_geometry(self) -> None:
        scene = scene_payload()
        scene.update(
            {
                "foreground_app_id": "com.vendor.runtime",
                "screen_id": "sample_app_conversation",
                "summary": "示例应用当前页面可见",
                "elements": [
                    {
                        "element_id": "title",
                        "role": "text",
                        "meaning": "page_title",
                        "label": "当前会话",
                        "bounds": [300, 20, 700, 80],
                        "confidence": 1.0,
                        "states": {"goal_relevant": True, "fully_visible": True},
                        "evidence": ["顶部标题"],
                    },
                    {
                        "element_id": "unusable-bottom-control",
                        "role": "button",
                        "meaning": "unrelated_action",
                        "label": "其他操作",
                        "bounds": [780, 950, 960, 1040],
                        "confidence": 1.0,
                        "states": {"goal_relevant": True, "fully_visible": True},
                        "evidence": ["底部控件"],
                    },
                ],
            }
        )
        provider = SequenceProvider(
            [
                {
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    "coordinate_space": {
                        "kind": "axis_grid",
                        "width": 1000,
                        "height": 960,
                    },
                    "scene": scene,
                    "input_structure": None,
                }
            ]
        )

        observed = SingleStepGenericSceneObserver(provider).observe(
            frames=stable_frames(),
            goal_context={
                "app_id": "sample_app",
                "app_name": "示例应用",
                "entities": {
                    "active_subgoal_visual_context": {
                        "subgoal_id": "launch_sample_app",
                        "objective": "打开示例应用",
                        "constraints": [],
                        "completion_conditions": ["示例应用已打开"],
                        "execution_class": "navigate",
                        "goal_entities": {},
                    }
                },
            },
            device_id="device-local-01",
        )

        self.assertEqual(1, provider.calls)
        self.assertEqual(["title"], [item.element_id for item in observed.elements])

    def test_single_step_input_revokes_unselected_overflow_compact_geometry(self) -> None:
        scene = scene_payload()
        scene["elements"] = [
            {
                "element_id": "input-overflow",
                "role": "input",
                "meaning": "message_input",
                "label": "",
                "bounds": [80, 960, 720, 1030],
                "confidence": 1.0,
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": ["输入框"],
            }
        ]
        provider = SequenceProvider(
            [
                {
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    "coordinate_space": {
                        "kind": "normalized_1000",
                        "width": 1000,
                        "height": 1000,
                    },
                    "scene": scene,
                    "input_structure": input_audit_payload(application_inputs=[]),
                }
            ]
        )
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_message",
                    "objective": "在输入框输入abc",
                    "constraints": [],
                    "completion_conditions": ["输入框内容为abc"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "abc",
                        "active_input_field_id": "message_field",
                        "active_input_multiline": False,
                    },
                }
            }
        }

        observed = SingleStepGenericSceneObserver(provider).observe(
            frames=stable_frames(), goal_context=context, device_id="device-local-01")

        self.assertEqual(1, provider.calls)
        self.assertIsNone(unique_goal_element(observed))

    def test_same_device_fingerprint_and_goal_still_call_qwen_for_each_observation(self) -> None:
        class PlainSequenceProvider:
            configured = True

            def __init__(self, responses: list[dict]) -> None:
                self.responses = list(responses)
                self.calls = 0

            def status(self) -> dict:
                return {"configured": True, "model": "offline-sequence"}

            def _chat(self, messages, max_tokens, **_kwargs) -> str:
                self.calls += 1
                return json.dumps(
                    self.responses.pop(0),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )

        def current_scene_envelope() -> dict:
            return {
                "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                "coordinate_space": {
                    "kind": "axis_grid",
                    "width": 1000,
                    "height": 960,
                },
                "scene": scene_payload(),
                "input_structure": None,
                "decision": default_model_decision(),
            }

        provider = PlainSequenceProvider([current_scene_envelope(), current_scene_envelope()])
        observer = SingleStepGenericSceneObserver(provider)
        frames = stable_frames()
        context: dict = {}

        first = observer.observe(
            frames=frames,
            goal_context=context,
            device_id="device-live-a",
        )
        second = observer.observe(
            frames=frames,
            goal_context=context,
            device_id="device-live-a",
        )

        self.assertEqual(2, provider.calls)
        self.assertIsNot(first, second)
        self.assertEqual(1, observer.last_diagnostics["model_calls"])
        self.assertEqual("single_step_current_scene_observation", observer.last_diagnostics["strategy"])
