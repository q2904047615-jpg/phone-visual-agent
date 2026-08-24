from __future__ import annotations

import json
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageFilter

from generic_scene_observer import (
    AUDITED_SOFT_KEYBOARD_HIDDEN_EVIDENCE,
    FOREGROUND_APP_IDENTITY_AUDIT_VERSION,
    GenericSceneObserver,
    SingleStepGenericSceneObserver,
    SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
    ICON_CLUSTER_AUDIT_VERSION,
    INPUT_STRUCTURE_AUDIT_VERSION,
    SYSTEM_UI_AUDIT_VERSION,
    TARGETED_SCENE_DELTA_PROTOCOL_VERSION,
    POST_NAVIGATION_RESULT_COMPLETION_CONDITIONS,
    POST_NAVIGATION_RESULT_OBJECTIVE,
    POST_NAVIGATION_RESULT_OBSERVATION_PHASE,
)
from generic_scene_observer import (
    _MAX_JSON_STRUCTURAL_REPAIR_CANDIDATES,
    _MAX_JSON_STRUCTURAL_REPAIR_CHARS,
    _camera_layout_orientation,
    _can_use_stable_ocr_literal_bounds,
    _compact_prompt,
    _foreground_app_identity_audit_prompt,
    _fused_preliminary_input_attestation,
    _goal_requests_input,
    _goal_requests_keyboard_mode_switch,
    _input_audit_literal_key_targets,
    _input_audit_retry_roi,
    _input_structure_audit_prompt,
    _apply_input_structure_audit,
    _map_input_structure_crop_audit_to_full,
    _needs_targeted_refinement,
    _observation_goal_context,
    _stable_ocr_literal_bounds,
    _input_structure_diagnostic_shape,
    _parse_scene_after_unique_structural_edit,
    _parse_targeted_scene_delta,
    _parse_scene,
    _scene_enum_values,
    _safe_goal_context,
    _select_keyboard_layout_switch_for_target,
    _single_json_structural_edits,
    _snap_reload_audit_to_local_glyph,
    _strict_icon_cluster_audit_payload,
    _strict_foreground_app_identity_audit,
    _strip_preliminary_keyboard_containers_for_dedicated_audit,
    _strip_model_authored_local_attestations,
    _targeted_prompt,
    _validated_keyboard_layout_switches,
    _validated_keyboard_backspace_key,
    _validated_keyboard_literal_keys,
    _validated_keyboard_mode_switch,
)
from ocr_runtime import OcrMatch
from input_value_lineage import TYPED_INPUT_LINEAGE_VERSION, TypedInputLineage
from orientation_safety import ORIENTATION_AUDIT_PROTOCOL_VERSION
from ui_scene import UI_SCENE_PROTOCOL_VERSION, UISceneError
from vision_agent import VisionAgentError


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
    ) -> str:
        self.calls += 1
        self.messages = messages
        self.max_tokens = max_tokens
        self.call_options = {
            "timeout": timeout,
            "max_attempts": max_attempts,
        }
        payload = self.payload
        if (
            self.calls == 2
            and payload.get("protocol_version") == UI_SCENE_PROTOCOL_VERSION
        ):
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
    ) -> str:
        self.calls += 1
        self.messages_seen.append(messages)
        self.max_tokens_seen.append(max_tokens)
        self.call_options = {"timeout": timeout, "max_attempts": max_attempts}
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        # Existing scene fixtures describe the intended refined facts. Adapt
        # only the second model response to the production targeted-delta wire
        # contract so the large historical suite does not duplicate fixtures.
        if (
            self.calls == 2
            and isinstance(value, dict)
            and value.get("protocol_version") == UI_SCENE_PROTOCOL_VERSION
        ):
            value = targeted_delta_payload(
                elements=value.get("elements") or [],
                summary_addendum=str(value.get("summary") or "")[:120],
                confidence=float(value.get("confidence") or 0.0),
            )
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def stable_frames(color: tuple[int, int, int] = (30, 40, 50)) -> list[Image.Image]:
    return [Image.new("RGB", (540, 960), color) for _ in range(4)]


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
        and str(resolved_keyboard.get("layout") or "").strip().casefold()
        == "qwerty"
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
            cluster_bounds
            if cluster_bounds is not None
            else [700, 0, 920, 100]
            if resolved_controls
            else None
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
        "visible_editable_cues": list(
            ["完整横向输入边框"]
            if visible_editable_cues is None
            else visible_editable_cues
        ),
        "caret_line_index": caret_line_index,
        "confidence": confidence,
        "right_button": right_button,
    }
    if field_labels is not None:
        value["field_labels"] = list(field_labels)
    return value


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
        markers.update({
            "active_input_predecessor_field_id": source["field_id"],
            "active_input_predecessor_field_label": source["field_label"],
            "active_input_predecessor_text": source["text"],
        })
    return {"entities": {"input_fields": fields, "active_subgoal_visual_context": {
        "subgoal_id": "input_body", "objective": "输入下一字段",
        "constraints": [], "completion_conditions": [],
        "execution_class": "navigate", "goal_entities": markers,
    }}}


def multifield_next_base(fields: list[dict], *, duplicate: bool = False):
    source = next(item for item in fields if item["field_id"] == "subject_field")
    visible = {
        "element_id": "source-visible", "role": "input",
        "meaning": "application_text_input", "label": source["field_label"],
        "bounds": [120, 430, 880, 590], "confidence": 0.98,
        "states": {
            "goal_relevant": True, "fully_visible": True, "focused": True,
            "value": source["text"], "input_field_id": source["field_id"],
            "input_field_label": source["field_label"],
        },
        "evidence": [source["field_label"], "caret"],
    }
    payload = scene_payload()
    payload["elements"] = [visible]
    if duplicate:
        payload["elements"].append({
            **visible, "element_id": "source-duplicate",
            "bounds": [120, 250, 880, 400],
        })
    return _parse_scene(json.dumps(payload, ensure_ascii=False), fingerprint="f" * 64)


def multifield_next_audit(
    fields: list[dict], *, action: str = "next",
    fully_visible: bool = True, confidence: float = 0.98,
    duplicate: bool = False,
) -> dict:
    source = next(item for item in fields if item["field_id"] == "subject_field")
    application_inputs = [audited_application_input(
        structure_id="source", bounds=[120, 430, 880, 590],
        text=source["text"], field_labels=[source["field_label"]],
    )]
    if duplicate:
        application_inputs.append(audited_application_input(
            structure_id="source-duplicate", bounds=[120, 250, 880, 400],
            text=source["text"], field_labels=[source["field_label"]],
        ))
    return input_audit_payload(
        application_inputs=application_inputs,
        keyboard={
            "visible": True, "bounds": [0, 600, 1000, 1000],
            "layout": "qwerty", "input_mode": "direct_latin",
            "case_mode": "lower", "mode_switch": None,
            "enter_key": {
                "label": "下一步", "bounds": [820, 920, 990, 985],
                "confidence": confidence, "fully_visible": fully_visible,
                "key_action": action,
            },
        },
    )


class GenericSceneObserverTests(unittest.TestCase):
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
        self.assertEqual(
            observed.unique_trusted_goal_element().element_id,
            "local_audited_input_1",
        )

    def test_single_step_observer_accepts_one_fused_blank_input_without_placeholder(
        self,
    ) -> None:
        scene = scene_payload()
        scene.update(
            {
                "foreground_app_id": "com.example.messaging",
                "screen_id": "named_conversation",
                "summary": "指定会话页底部有一个空输入框",
                "elements": [
                    {
                        "element_id": "e1",
                        "role": "input",
                        "meaning": "message_input_field",
                        "label": "",
                        "bounds": [120, 910, 780, 960],
                        "confidence": 1.0,
                        "states": {
                            "goal_relevant": True,
                            "fully_visible": True,
                            "value": "",
                        },
                        "evidence": ["底部工具栏中唯一完整白色文本输入区域"],
                    }
                ],
            }
        )
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="message",
                    bounds=[120, 910, 780, 960],
                    text="",
                    placeholder="",
                    visible_editable_cues=[],
                )
            ]
        )
        provider = SequenceProvider(
            [
                {
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    "scene": scene,
                    "input_structure": audit,
                }
            ]
        )
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

        observed = SingleStepGenericSceneObserver(provider).observe(
            frames=stable_frames(),
            goal_context=context,
            device_id="device-local-01",
        )

        self.assertEqual(provider.calls, 1)
        field = observed.unique_trusted_goal_element()
        self.assertEqual("local_audited_input_1", field.element_id)
        self.assertEqual("", field.states["value"])
        self.assertIn("当前输入框为空", " ".join(field.evidence))

    def test_single_step_observer_restores_only_missing_nested_audit_version(self) -> None:
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
        self.assertEqual(
            observed.unique_trusted_goal_element().element_id,
            "local_audited_input_1",
        )

    def test_single_step_observer_rejects_other_nested_audit_shape_changes(self) -> None:
        scene = scene_payload()
        audit = input_audit_payload(application_inputs=[])
        audit.pop("protocol_version")
        audit["unexpected"] = True
        provider = SequenceProvider(
            [
                {
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
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

        with self.assertRaisesRegex(VisionAgentError, "协议外字段"):
            SingleStepGenericSceneObserver(provider).observe(
                frames=stable_frames(),
                goal_context=context,
                device_id="device-local-01",
            )

        self.assertEqual(provider.calls, 1)

    def test_single_step_observer_does_not_retry_malformed_response(self) -> None:
        provider = SequenceProvider(["{not-json"])
        observer = SingleStepGenericSceneObserver(provider)

        with self.assertRaises(VisionAgentError):
            observer.observe(frames=stable_frames())

        self.assertEqual(provider.calls, 1)
        self.assertEqual(observer.last_diagnostics["model_calls"], 1)
        self.assertFalse(observer.last_diagnostics["remote_retry_used"])

    def test_single_step_observer_accepts_flat_non_input_scene_in_one_call(self) -> None:
        provider = SequenceProvider([scene_payload()])
        observer = SingleStepGenericSceneObserver(provider)

        observed = observer.observe(frames=stable_frames())

        self.assertEqual(provider.calls, 1)
        self.assertEqual(observed.foreground_app_id, "calculator")
        self.assertEqual(observer.last_diagnostics["model_calls"], 1)
        self.assertEqual(
            observer.last_diagnostics["online_stages"],
            ["single_step_observation"],
        )

    def test_exact_device_fingerprint_and_goal_context_reuse_observation(self) -> None:
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

        provider = PlainSequenceProvider([scene_payload(), scene_payload()])
        observer = GenericSceneObserver(provider)
        frames = stable_frames()
        context: dict = {}

        first = observer.observe(
            frames=frames,
            goal_context=context,
            device_id="device-cache-a",
        )
        second = observer.observe(
            frames=frames,
            goal_context=context,
            device_id="device-cache-a",
        )

        self.assertIs(first, second)
        self.assertEqual(1, provider.calls)
        self.assertTrue(observer.last_diagnostics["observation_cache_hit"])
        self.assertEqual(0, observer.last_diagnostics["model_calls"])

        observer.observe(
            frames=frames,
            goal_context=context,
            device_id="device-cache-b",
        )
        self.assertEqual(2, provider.calls)

    def test_pending_typed_lineage_reuses_only_surface_identity_then_reaudits_input(
        self,
    ) -> None:
        first_compact = scene_payload()
        first_compact["elements"] = [
            {
                "element_id": "model-input",
                "role": "input",
                "meaning": "text_input_field",
                "label": "",
                "bounds": [140, 270, 860, 450],
                "confidence": 0.98,
                "states": {
                    "goal_relevant": True,
                    "fully_visible": True,
                    "focused": True,
                    "value": "a",
                },
                "evidence": ["正文输入框和光标可见"],
            }
        ]
        first_audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="body",
                    bounds=[140, 270, 860, 450],
                    text="a",
                    field_labels=["正文"],
                    caret_line_index=0,
                )
            ],
            keyboard={
                "visible": True,
                "bounds": [80, 570, 920, 1000],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "case_mode": "lower",
                "mode_switch": None,
            },
        )
        second_audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="body",
                    bounds=[140, 270, 860, 450],
                    text="ab",
                    field_labels=["正文"],
                    caret_line_index=0,
                )
            ],
            keyboard={
                "visible": True,
                "bounds": [80, 570, 920, 1000],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "case_mode": "lower",
                "mode_switch": None,
            },
        )
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input-body",
                    "objective": "在正文输入ab",
                    "constraints": ["不得发送"],
                    "completion_conditions": ["正文逐字等于ab"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "input_text": "ab",
                        "active_input_transaction_text": "ab",
                        "active_input_field_id": "body_field",
                        "active_input_field_label": "正文",
                        "active_input_multiline": False,
                    },
                }
            }
        }
        provider = SequenceProvider(
            [first_compact, first_audit, second_audit, second_audit]
        )
        observer = GenericSceneObserver(provider)
        first = observer.observe(
            frames=stable_frames((30, 40, 50)),
            goal_context=context,
            device_id="device-continuation",
        )
        self.assertEqual(2, provider.calls)

        lineage = TypedInputLineage(
            version=TYPED_INPUT_LINEAGE_VERSION,
            device_id="device-continuation",
            exact_value="ab",
            app_id=first.app_id,
            screen_id=first.screen_id,
            input_meaning="application_text_input",
            input_field_id="body_field",
            input_bounds=(0.14, 0.27, 0.86, 0.45),
            before_fingerprint=first.fingerprint,
            after_fingerprint="pending-new-fingerprint",
            action_digest="a" * 64,
            receipt_digest="b" * 64,
            surface_descriptors=(),
            recorded_at_epoch=time.time(),
            source="pending_verified_text_action",
        )
        lineage.validate()
        continued = observer.observe(
            frames=stable_frames((50, 60, 70)),
            goal_context=context,
            device_id="device-continuation",
            input_lineage_override=lineage,
            prior_scene=first,
        )

        self.assertEqual(3, provider.calls)
        self.assertTrue(
            observer.last_diagnostics["compact_reused_from_typed_lineage"]
        )
        self.assertTrue(observer.last_diagnostics["input_structure_audit_used"])
        self.assertEqual(1, observer.last_diagnostics["model_calls"])
        self.assertEqual("ab", continued.get_element("local_audited_input_1").states["value"])

        chinese_preedit_lineage = replace(
            lineage,
            source="pending_verified_chinese_preedit_action",
            action_digest="c" * 64,
            receipt_digest="d" * 64,
        )
        chinese_preedit_lineage.validate()
        continued_after_preedit = observer.observe(
            frames=stable_frames((70, 80, 90)),
            goal_context=context,
            device_id="device-continuation",
            input_lineage_override=chinese_preedit_lineage,
            prior_scene=first,
        )

        self.assertEqual(4, provider.calls)
        self.assertTrue(
            observer.last_diagnostics["compact_reused_from_typed_lineage"]
        )
        self.assertEqual(1, observer.last_diagnostics["model_calls"])
        self.assertEqual(
            "ab",
            continued_after_preedit.get_element("local_audited_input_1").states[
                "value"
            ],
        )

    def test_goal_context_still_rejects_genuinely_excessive_nesting(self) -> None:
        context = {
            "level_1": {
                "level_2": {
                    "level_3": {
                        "level_4": {"level_5": {"level_6": "too deep"}}
                    }
                }
            }
        }

        with self.assertRaisesRegex(VisionAgentError, "目标上下文嵌套过深"):
            _safe_goal_context(context)

    def test_verified_navigation_result_uses_one_compact_coordinate_space(self) -> None:
        payload = scene_payload()
        payload.update(
            {
                "foreground_app_id": "browser",
                "screen_id": "unknown",
                "summary": "浏览器结果页底部有主页和窗口标签。",
                "elements": [],
                "confidence": 0.98,
            }
        )
        context = {
            "app_id": "browser",
            "app_name": "浏览器",
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "open_browser",
                    "objective": POST_NAVIGATION_RESULT_OBJECTIVE,
                    "constraints": ["仅导航"],
                    "completion_conditions": list(
                        POST_NAVIGATION_RESULT_COMPLETION_CONDITIONS
                    ),
                    "execution_class": "navigate",
                    "goal_entities": {
                        "target_surface": "device",
                        "observation_phase": (
                            POST_NAVIGATION_RESULT_OBSERVATION_PHASE
                        ),
                    },
                }
            },
        }
        provider = SequenceProvider([payload])
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=stable_frames(),
            goal_context=context,
        )

        self.assertEqual("browser", scene.foreground_app_id)
        self.assertEqual(1, provider.calls)
        self.assertFalse(observer.last_diagnostics["targeted_refinement_used"])
        self.assertFalse(_needs_targeted_refinement(scene, context))

        unmarked = json.loads(json.dumps(context, ensure_ascii=False))
        unmarked_focus = unmarked["entities"]["active_subgoal_visual_context"]
        unmarked_focus["objective"] = "打开浏览器"
        unmarked_focus["completion_conditions"] = ["浏览器结果页可见"]
        unmarked_focus["goal_entities"].pop("observation_phase")
        self.assertTrue(_needs_targeted_refinement(scene, unmarked))

    def test_input_audit_prompt_defines_exact_nullable_mode_switch_contract(self) -> None:
        prompt = _input_structure_audit_prompt(
            {"objective": "切换当前键盘输入模式"},
            roi_bounds=None,
        )

        self.assertIn(INPUT_STRUCTURE_AUDIT_VERSION, prompt)
        self.assertIn('"mode_switch":null', prompt)
        self.assertIn(
            '{"label":"英","bounds":[0,0,1000,1000],"confidence":0.0,'
            '"current_mode":"chinese_pinyin","target_mode":"direct_latin"}',
            prompt,
        )
        self.assertIn(
            '{"label":"英","bounds":[0,0,1000,1000],"confidence":0.0,'
            '"current_mode":"direct_latin","target_mode":"chinese_pinyin"}',
            prompt,
        )
        self.assertIn(
            "visible key label may name either the current mode or the "
            "destination mode",
            prompt,
        )
        self.assertIn(
            "never derive current_mode or target_mode from that label",
            prompt,
        )
        self.assertIn("Never omit confidence or target_mode", prompt)
        self.assertIn(
            "A complete blank surface does not need placeholder text, a caret, "
            "or focus highlight",
            prompt,
        )
        self.assertIn(
            "Never infer a field from the goal, an unexplained gap, or adjacent "
            "icons alone",
            prompt,
        )
        self.assertNotIn(
            "Include an empty field only when a complete border plus a visible "
            "placeholder",
            prompt,
        )
        self.assertIn("compact minified", prompt)
        self.assertIn("single line", prompt)
        self.assertIn("the local, goal-derived whitelist is []", prompt)
        self.assertIn("Never enumerate a keyboard row", prompt)
        self.assertIn(
            "the bottom edge is always 1000, never a source-pixel or "
            "conventional display height",
            prompt,
        )
        self.assertIn("has width at least 300 and height at least 180", prompt)
        self.assertIn("contains every reported keyboard key and anchor", prompt)
        self.assertIn("set bounds=null instead of inventing it", prompt)
        self.assertIn(
            "only after independent multi-frame row evidence validates all seven anchors",
            prompt,
        )
        crop_prompt = _input_structure_audit_prompt(
            {"objective": "在正文输入框输入两行文字"},
            roi_bounds=(0, 130, 1000, 1000),
            crop_local=True,
        )
        self.assertIn(
            "at least 15 units of crop-local margin from that edge",
            crop_prompt,
        )
        self.assertEqual(
            (" ", "."),
            _input_audit_literal_key_targets(
                {"entities": {"input_text": "draft message."}}
            ),
        )
        mixed_context = {
            "entities": {"input_text": "复杂输入验收2026:123+45"}
        }
        self.assertEqual(
            ("1",),
            _input_audit_literal_key_targets(
                mixed_context,
                current_input_text="复杂输入验收2026:",
            ),
        )
        self.assertEqual(
            ("2",),
            _input_audit_literal_key_targets(
                mixed_context,
                current_input_text="复杂输入验收2026:1",
            ),
        )
        self.assertEqual(
            ("5",),
            _input_audit_literal_key_targets(
                {"entities": {"input_text": "复杂输入验收2026:123+45-6@7."}},
                current_input_text="复杂输入验收2026:123+4",
            ),
        )
        self.assertEqual(
            (),
            _input_audit_literal_key_targets(
                {"entities": {"input_text": "复杂输入验收2026:长"}},
                current_input_text="复杂输入验收2026:",
            ),
        )
        self.assertEqual(
            ("2", "0", "6", ":", "1", "3", "+", "4"),
            _input_audit_literal_key_targets(
                mixed_context,
                current_input_text="不一致前缀",
            ),
        )
        direct_prompt = _input_structure_audit_prompt(
            {"entities": {"input_text": "wifi"}},
            roi_bounds=None,
        )
        self.assertIn("the local, goal-derived whitelist is []", direct_prompt)

        literal_prompt = _input_structure_audit_prompt(
            {"entities": {"input_text": "复杂输入验收2026:123+45-6@7."}},
            roi_bounds=None,
            current_input_text="复杂输入验收2026:123+45-6",
        )
        self.assertIn('the local, goal-derived whitelist is ["@"]', literal_prompt)
        self.assertIn(
            '"literal_keys":[{"value":"@","label":"@",'
            '"key_kind":"character","bounds":[0,0,1000,1000],'
            '"confidence":0.0,"fully_visible":true}]',
            literal_prompt,
        )
        self.assertIn(
            "Every literal-key object MUST contain exactly these six fields",
            literal_prompt,
        )
        self.assertIn(
            "A small corner glyph, superscript digit, alternate symbol, swipe hint "
            "or long-press hint printed on an alphabet key is NOT a literal key",
            literal_prompt,
        )
        self.assertIn(
            "Bounds must enclose the whole direct key, never only the secondary glyph",
            literal_prompt,
        )
        self.assertIn(
            "An automatic visual line wrap inside a narrow editable field is "
            "presentation only",
            literal_prompt,
        )

    def test_observer_does_not_use_compact_prefix_as_literal_key_authority(
        self,
    ) -> None:
        preliminary = scene_payload()
        preliminary["elements"] = [
            {
                "element_id": "model-input",
                "role": "input",
                "meaning": "application_text_input",
                "label": "复杂输入验收2026:",
                "bounds": [150, 530, 690, 590],
                "confidence": 0.98,
                "states": {
                    "goal_relevant": True,
                    "fully_visible": True,
                    "value": "复杂输入验收2026:",
                    "focused": True,
                    "keyboard_layout": "numeric",
                    "keyboard_input_mode": "unknown",
                },
                "evidence": ["输入框逐字显示当前前缀"],
            }
        ]
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    text="复杂输入验收2026:",
                    bounds=[150, 530, 690, 590],
                )
            ],
            keyboard={
                "visible": True,
                "bounds": [0, 600, 1000, 1000],
                "layout": "numeric",
                "input_mode": "unknown",
                "mode_switch": None,
                "literal_keys": [
                    {
                        "value": "1",
                        "label": "1",
                        "key_kind": "character",
                        "bounds": [180, 690, 340, 750],
                        "confidence": 1.0,
                        "fully_visible": True,
                    }
                ],
            },
        )
        provider = SequenceProvider([preliminary, audit])

        scene = GenericSceneObserver(provider).observe(
            frames=stable_frames(),
            goal_context={
                "objective": "输入框最终逐字显示复杂输入验收2026:123+45",
                "entities": {"input_text": "复杂输入验收2026:123+45"},
            },
        )

        prompt = provider.messages_seen[1][1]["content"][0]["text"]
        self.assertIn(
            'the local, goal-derived whitelist is ["2","0","6",":","1","3","+","4"]',
            prompt,
        )
        self.assertNotIn(
            'the local, goal-derived whitelist is ["1"]',
            prompt,
        )
        self.assertEqual(
            "1",
            scene.unique_trusted_goal_element().states["key_value"],
        )

    def test_literal_ocr_geometry_is_limited_to_text_bearing_selector_roles(self) -> None:
        for role in ("text", "button", "tab", "list_item"):
            self.assertTrue(_can_use_stable_ocr_literal_bounds(role, "入口"))
        for role in ("input", "icon", "image", "container"):
            self.assertFalse(_can_use_stable_ocr_literal_bounds(role, "入口"))
        self.assertFalse(_can_use_stable_ocr_literal_bounds("button", "  "))

    def test_clickable_literal_geometry_prefers_unique_three_frame_ocr_consensus(self) -> None:
        frames = [Image.new("RGB", (810, 1440), "black") for _ in range(3)]
        results = iter(
            [
                [OcrMatch("返回验收模式选择", 93, 132, 225, 31)],
                [OcrMatch("返回验收模式选择", 94, 131, 225, 31)],
                [OcrMatch("返回验收模式选择", 93, 132, 226, 31)],
            ]
        )

        bounds = _stable_ocr_literal_bounds(
            frames,
            "返回验收模式选择",
            ocr_recognizer=lambda *_args, **_kwargs: {},
            ocr_finder=lambda *_args, **_kwargs: next(results),
        )

        self.assertIsNotNone(bounds)
        assert bounds is not None
        self.assertAlmostEqual(93 / 810, bounds[0])
        self.assertAlmostEqual(132 / 1440, bounds[1])
        self.assertAlmostEqual(319 / 810, bounds[2])
        self.assertAlmostEqual(163 / 1440, bounds[3])

    def test_literal_text_geometry_rejects_duplicate_or_unstable_ocr(self) -> None:
        frames = [Image.new("RGB", (810, 1440), "black") for _ in range(3)]
        duplicate_results = iter(
            [
                [
                    OcrMatch("返回验收模式选择", 93, 132, 225, 31),
                    OcrMatch("返回验收模式选择", 400, 600, 225, 31),
                ],
                [OcrMatch("返回验收模式选择", 93, 132, 225, 31)],
                [OcrMatch("返回验收模式选择", 93, 132, 225, 31)],
            ]
        )
        self.assertIsNone(
            _stable_ocr_literal_bounds(
                frames,
                "返回验收模式选择",
                ocr_recognizer=lambda *_args, **_kwargs: {},
                ocr_finder=lambda *_args, **_kwargs: next(duplicate_results),
            )
        )

        unstable_results = iter(
            [
                [OcrMatch("返回验收模式选择", 93, 132, 225, 31)],
                [OcrMatch("返回验收模式选择", 94, 131, 225, 31)],
                [OcrMatch("返回验收模式选择", 93, 150, 225, 31)],
            ]
        )
        self.assertIsNone(
            _stable_ocr_literal_bounds(
                frames,
                "返回验收模式选择",
                ocr_recognizer=lambda *_args, **_kwargs: {},
                ocr_finder=lambda *_args, **_kwargs: next(unstable_results),
            )
        )

    def test_compact_prompt_forbids_copying_source_pixel_coordinates(self) -> None:
        prompt = _compact_prompt({"objective": "读取当前页面"})
        self.assertIn("禁止复制原图像素坐标", prompt)
        self.assertIn("810x1515", prompt)
        self.assertIn("任何边界超出0..1000就省略该元素", prompt)

    def test_observation_prompts_define_strict_scrollable_viewport_fact(self) -> None:
        context = {"objective": "在当前列表向上滑动一次"}
        compact = _compact_prompt(context)
        targeted = _targeted_prompt(
            context,
            first_scene={
                "foreground_app_id": "unknown",
                "screen_id": "list",
                "summary": "列表可见",
                "system_ui": {},
                "overlays": [],
                "confidence": 1.0,
            },
        )
        for prompt in (compact, targeted):
            self.assertIn("scrollable:true", prompt)
            self.assertIn('scroll_axis:"vertical"或"horizontal"', prompt)
            self.assertIn("至少两个", prompt)
            self.assertIn("单张卡片", prompt)

    def test_input_observation_prompts_exclude_regular_keys_from_compact_budget(self) -> None:
        context = {"objective": "让当前唯一空白输入框显示 wifi，不提交"}
        compact = _compact_prompt(context)
        targeted = _targeted_prompt(
            context,
            first_scene={
                "foreground_app_id": "unknown",
                "screen_id": "search",
                "summary": "输入页",
                "system_ui": {},
                "overlays": [],
                "confidence": 1.0,
            },
        )

        for prompt in (compact, targeted):
            self.assertIn("普通键不得进入elements", prompt)
            self.assertIn("独立全帧输入结构审计负责", prompt)
            self.assertIn("role=button", prompt)
            self.assertNotIn("普通键仍必须role=keyboard_key", prompt)

    def test_observation_prompts_preserve_only_observe_clipped_list_cue(self) -> None:
        compact = _compact_prompt({"objective": "查看目标结果"})
        targeted = _targeted_prompt(
            {"objective": "查看目标结果"},
            first_scene={
                "foreground_app_id": "unknown",
                "screen_id": "unknown",
                "summary": "当前列表",
                "system_ui": {},
                "overlays": [],
                "confidence": 1.0,
            },
        )

        for prompt in (compact, targeted):
            self.assertIn("部分可见的后续", prompt)
            self.assertIn("连续引导轨", prompt)
            self.assertIn("页面延续标记", prompt)
            self.assertIn("summary", prompt)
            self.assertIn("不得", prompt)
            self.assertIn("可操作目标", prompt)

    def test_out_of_range_explicit_non_goal_peripheral_is_discarded(self) -> None:
        payload = scene_payload()
        payload["elements"][0]["states"] = {"goal_relevant": True}
        payload["elements"].append(
            {
                "element_id": "pixel-coordinate-key",
                "role": "keyboard_key",
                "meaning": "keyboard_enter",
                "label": "开始",
                "bounds": [730, 1130, 810, 1190],
                "confidence": 0.99,
                "states": {"goal_relevant": False, "enabled": True},
                "evidence": ["键盘右下角按键"],
            }
        )

        scene = _parse_scene(
            json.dumps(payload, ensure_ascii=False),
            fingerprint="local-fingerprint",
            goal_context={"objective": "查看数字七"},
            camera_layout_orientation="portrait",
        )

        self.assertEqual(["e1"], [item.element_id for item in scene.elements])

    def test_out_of_range_goal_or_input_element_remains_fail_closed(self) -> None:
        for role, goal_relevant in (("button", True), ("input", False)):
            with self.subTest(role=role, goal_relevant=goal_relevant):
                payload = scene_payload()
                payload["elements"] = [
                    {
                        "element_id": "unsafe-pixel-coordinate",
                        "role": role,
                        "meaning": "target_control",
                        "label": "目标",
                        "bounds": [730, 1130, 810, 1190],
                        "confidence": 0.99,
                        "states": {"goal_relevant": goal_relevant},
                        "evidence": ["目标控件"],
                    }
                ]
                with self.assertRaisesRegex(VisionAgentError, "bounds"):
                    _parse_scene(
                        json.dumps(payload, ensure_ascii=False),
                        fingerprint="local-fingerprint",
                        goal_context=(
                            {
                                "objective": "在当前输入框输入 codex",
                                "entities": {"input_text": "codex"},
                            }
                            if role == "input"
                            else {"objective": "操作目标控件"}
                        ),
                        camera_layout_orientation="portrait",
                    )

    def test_keyboard_switch_goal_defers_one_invalid_compact_box_to_strict_audit(self) -> None:
        compact = scene_payload()
        compact["screen_id"] = "input_page"
        compact["summary"] = "输入框与软键盘可见"
        compact["elements"] = [
            {
                "element_id": "pixel-coordinate-mode-key",
                "role": "button",
                "meaning": "switch_keyboard_input_mode",
                "label": "中",
                "bounds": [730, 1130, 830, 1210],
                "confidence": 0.99,
                "states": {
                    "goal_relevant": True,
                    "fully_visible": True,
                    "keyboard_input_mode_switch": True,
                    "current_mode": "chinese_pinyin",
                    "target_mode": "direct_latin",
                },
                "evidence": ["键盘底部模式键"],
            }
        ]
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(text="", placeholder="")
            ],
            keyboard={
                "visible": True,
                "bounds": [0, 360, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "chinese_pinyin",
                "mode_switch": {
                    "label": "中",
                    "bounds": [650, 900, 760, 970],
                    "confidence": 0.97,
                    "current_mode": "chinese_pinyin",
                    "target_mode": "direct_latin",
                },
            },
        )
        audit["application_inputs"][0]["visible_editable_cues"] = ["caret"]
        provider = SequenceProvider([compact, audit])

        observer = GenericSceneObserver(provider)
        scene = observer.observe(
            frames=stable_frames(),
            goal_context={"objective": "把当前键盘切换到英文直输模式"},
        )

        target = scene.unique_trusted_goal_element()
        self.assertIsNotNone(target)
        self.assertEqual("switch_keyboard_input_mode", target.meaning)
        self.assertEqual((0.65, 0.9, 0.76, 0.97), target.bounds)
        self.assertEqual(2, provider.calls)
        self.assertFalse(observer.last_diagnostics["targeted_refinement_used"])
        compact_prompt = provider.messages_seen[0][1]["content"][0]["text"]
        self.assertIn("独立全帧输入结构审计是模式、方向和模式键几何的唯一权威", compact_prompt)
        self.assertNotIn("必须另建role=button元素", compact_prompt)

    def test_exact_tap_restores_original_keyboard_mode_visual_context(self) -> None:
        compact = scene_payload()
        compact["screen_id"] = "input_page"
        compact["summary"] = "空输入框和QWERTY软键盘可见"
        compact["elements"] = []
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(text="", placeholder="")
            ],
            keyboard={
                "visible": True,
                "bounds": [0, 360, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "chinese_pinyin",
                "mode_switch": {
                    "label": "英",
                    "bounds": [650, 900, 760, 970],
                    "confidence": 0.97,
                    "current_mode": "chinese_pinyin",
                    "target_mode": "direct_latin",
                },
            },
        )
        audit["application_inputs"][0]["visible_editable_cues"] = ["caret"]
        provider = SequenceProvider([compact, audit])
        context = {
            "objective": "点击当前画面中的目标控件",
            "entities": {
                "target_ui_label": "英",
                "original_goal_visual_context": (
                    "只点击当前软键盘上逐字显示为英的输入模式切换键一次，"
                    "使键盘进入英文直输状态"
                ),
                "active_subgoal_visual_context": {
                    "subgoal_id": "exact_tap_semantic",
                    "objective": "点击当前画面中的目标控件",
                    "constraints": [],
                    "completion_conditions": ["动作后出现新的稳定画面"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "target_surface": "current_surface",
                        "target_ui_label": "英",
                    },
                },
            },
        }

        scene = GenericSceneObserver(provider).observe(
            frames=stable_frames(),
            goal_context=context,
        )

        target = scene.unique_trusted_goal_element()
        self.assertIsNotNone(target)
        self.assertEqual("local_audited_keyboard_mode_switch_1", target.element_id)
        self.assertEqual("英", target.label)
        self.assertEqual("chinese_pinyin", target.states["current_mode"])
        self.assertEqual("direct_latin", target.states["target_mode"])
        self.assertEqual(2, provider.calls)
        self.assertTrue(_goal_requests_keyboard_mode_switch(context))
        self.assertTrue(_goal_requests_input(context))

    def test_normal_subgoal_ignores_root_future_keyboard_mode_text(self) -> None:
        context = {
            "objective": "打开页面后切换到英文直输模式",
            "entities": {
                "original_goal_visual_context": "打开页面后切换到英文直输模式",
                "active_subgoal_visual_context": {
                    "subgoal_id": "open_page",
                    "objective": "打开当前页面",
                    "constraints": [],
                    "completion_conditions": ["页面已打开"],
                    "execution_class": "navigate",
                    "goal_entities": {"target_ui_label": "打开"},
                },
            },
        }

        self.assertFalse(_goal_requests_keyboard_mode_switch(context))
        self.assertFalse(_goal_requests_input(context))

    def test_unrelated_exact_tap_does_not_trigger_keyboard_input_audit(self) -> None:
        context = {
            "objective": "点击当前画面中的目标控件",
            "entities": {
                "original_goal_visual_context": "只点击逐字显示为刷新的控件一次",
                "active_subgoal_visual_context": {
                    "subgoal_id": "exact_tap_semantic",
                    "objective": "点击当前画面中的目标控件",
                    "constraints": [],
                    "completion_conditions": ["动作后出现新的稳定画面"],
                    "execution_class": "navigate",
                    "goal_entities": {"target_ui_label": "刷新"},
                },
            },
        }

        self.assertFalse(_goal_requests_keyboard_mode_switch(context))
        self.assertFalse(_goal_requests_input(context))

    def test_keyboard_switch_goal_defers_all_safe_preliminary_shapes_to_strict_audit(self) -> None:
        variants = (
            (
                "keyboard_key",
                "英",
                [680, 1130, 790, 1210],
                {
                    "goal_relevant": False,
                    "keyboard_input_mode_switch": True,
                },
            ),
            (
                "icon",
                "",
                [730, 1130, 810, 1200],
                {"goal_relevant": True},
            ),
            (
                "button",
                "英",
                [730, 890, 810, 940],
                {
                    "goal_relevant": True,
                    "keyboard_input_mode_switch": True,
                    "current_mode": "direct_latin",
                    "target_mode": "chinese_pinyin",
                },
            ),
            (
                "input",
                "",
                [100, 80, 900, 1130],
                {
                    "goal_relevant": True,
                    "value": "",
                    "focused": True,
                },
            ),
        )
        for role, label, bounds, states in variants:
            with self.subTest(role=role, states=states):
                compact = scene_payload()
                compact["screen_id"] = "input_page"
                compact["summary"] = "输入框与软键盘可见"
                compact["elements"] = [
                    {
                        "element_id": "preliminary-mode-key",
                        "role": role,
                        "meaning": "switch_keyboard_input_mode",
                        "label": label,
                        "bounds": bounds,
                        "confidence": 0.99,
                        "states": states,
                        "evidence": ["普通场景初步看到模式键"],
                    }
                ]
                audit = input_audit_payload(
                    application_inputs=[
                        audited_application_input(text="", placeholder="")
                    ],
                    keyboard={
                        "visible": True,
                        "bounds": [0, 360, 1000, 1000],
                        "layout": "qwerty",
                        "input_mode": "chinese_pinyin",
                        "mode_switch": {
                            "label": "中/英",
                            "bounds": [650, 900, 760, 970],
                            "confidence": 0.97,
                            "current_mode": "chinese_pinyin",
                            "target_mode": "direct_latin",
                        },
                    },
                )
                audit["application_inputs"][0]["visible_editable_cues"] = ["caret"]
                provider = SequenceProvider([compact, audit])

                scene = GenericSceneObserver(provider).observe(
                    frames=stable_frames(),
                    goal_context={"objective": "把当前键盘切换到英文直输模式"},
                )

                target = scene.unique_trusted_goal_element()
                self.assertEqual(
                    "local_audited_keyboard_mode_switch_1",
                    target.element_id,
                )
                self.assertEqual((0.65, 0.9, 0.76, 0.97), target.bounds)
                self.assertEqual("chinese_pinyin", target.states["current_mode"])
                self.assertEqual("direct_latin", target.states["target_mode"])
                self.assertEqual(2, provider.calls)

    def test_keyboard_switch_goal_does_not_hide_invalid_strict_audit_geometry(self) -> None:
        compact = scene_payload()
        compact["summary"] = "输入框与软键盘可见"
        compact["elements"] = [
            {
                "element_id": "preliminary-mode-key",
                "role": "keyboard_key",
                "meaning": "switch_keyboard_input_mode",
                "label": "英",
                "bounds": [730, 1130, 810, 1210],
                "confidence": 0.99,
                "states": {"goal_relevant": True},
                "evidence": ["普通场景初步看到模式键"],
            }
        ]
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(text="", placeholder="")
            ],
            keyboard={
                "visible": True,
                "bounds": [0, 360, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "chinese_pinyin",
                "mode_switch": {
                    "label": "中/英",
                    "bounds": [680, 1130, 790, 1210],
                    "confidence": 0.97,
                    "current_mode": "chinese_pinyin",
                    "target_mode": "direct_latin",
                },
            },
        )
        audit["application_inputs"][0]["visible_editable_cues"] = ["caret"]

        with self.assertRaisesRegex(VisionAgentError, "mode_switch bounds 无效"):
            GenericSceneObserver(
                SequenceProvider([compact, audit])
            ).observe(
                frames=stable_frames(),
                goal_context={"objective": "把当前键盘切换到英文直输模式"},
            )

    def test_malformed_invalid_keyboard_switch_is_not_hidden_by_audit_deferral(self) -> None:
        compact = scene_payload()
        compact["elements"] = [
            {
                "element_id": "unsafe-mode-key",
                "role": "button",
                "meaning": "switch_keyboard_input_mode",
                "label": "中",
                "bounds": [730, 1130, 830, 1210],
                "confidence": 0.99,
                "states": {
                    "goal_relevant": True,
                    "keyboard_input_mode_switch": True,
                    "current_mode": "chinese_pinyin",
                    "target_mode": "direct_latin",
                    "action": "tap",
                },
                "evidence": ["键盘底部模式键"],
            }
        ]

        with self.assertRaises(VisionAgentError):
            GenericSceneObserver(SequenceProvider([compact])).observe(
                frames=stable_frames(),
                goal_context={"objective": "把当前键盘切换到英文直输模式"},
            )

    def test_invalid_model_keyboard_mode_switch_claim_is_revoked(self) -> None:
        for current_mode, target_mode in (
            ("symbol", "unknown"),
            ("direct_latin", "direct_latin"),
        ):
            with self.subTest(
                current_mode=current_mode,
                target_mode=target_mode,
            ):
                payload = scene_payload()
                payload["elements"] = [
                    {
                        "element_id": "model-symbol-key",
                        "role": "button",
                        "meaning": "switch_keyboard_input_mode",
                        "label": "符",
                        "bounds": [60, 890, 200, 960],
                        "confidence": 1.0,
                        "states": {
                            "goal_relevant": False,
                            "fully_visible": True,
                            "keyboard_input_mode_switch": True,
                            "current_mode": current_mode,
                            "target_mode": target_mode,
                        },
                        "evidence": ["键盘左下角可见符号键"],
                    }
                ]

                _strip_model_authored_local_attestations(payload)

                states = payload["elements"][0]["states"]
                self.assertNotIn("keyboard_input_mode_switch", states)
                self.assertNotIn("current_mode", states)
                self.assertNotIn("target_mode", states)
                self.assertEqual("符", payload["elements"][0]["label"])
                self.assertEqual(
                    "switch_keyboard_input_mode",
                    payload["elements"][0]["meaning"],
                )

    def test_invalid_model_keyboard_switch_claim_no_longer_blocks_scene_parse(
        self,
    ) -> None:
        payload = scene_payload()
        payload["elements"] = [
            {
                "element_id": "model-symbol-key",
                "role": "button",
                "meaning": "switch_keyboard_input_mode",
                "label": "符",
                "bounds": [60, 890, 200, 960],
                "confidence": 1.0,
                "states": {
                    "goal_relevant": False,
                    "fully_visible": True,
                    "keyboard_input_mode_switch": True,
                    "current_mode": "symbol",
                    "target_mode": "unknown",
                },
                "evidence": ["键盘左下角可见符号键"],
            }
        ]

        scene = _parse_scene(
            json.dumps(payload, ensure_ascii=False),
            fingerprint="invalid-model-switch-claim",
            goal_context={
                "objective": "在当前输入框继续输入2026",
                "entities": {"input_text": "2026"},
            },
            camera_layout_orientation="portrait",
        )

        states = scene.elements[0].states
        self.assertNotIn("keyboard_input_mode_switch", states)
        self.assertNotIn("current_mode", states)
        self.assertNotIn("target_mode", states)

    def test_valid_model_keyboard_mode_switch_direction_remains_strict(self) -> None:
        payload = scene_payload()
        payload["elements"] = [
            {
                "element_id": "model-language-key",
                "role": "button",
                "meaning": "switch_keyboard_input_mode",
                "label": "中",
                "bounds": [730, 890, 830, 960],
                "confidence": 1.0,
                "states": {
                    "goal_relevant": False,
                    "fully_visible": True,
                    "keyboard_input_mode_switch": True,
                    "current_mode": "chinese_pinyin",
                    "target_mode": "direct_latin",
                },
                "evidence": ["键盘底部可见语言模式键"],
            }
        ]

        _strip_model_authored_local_attestations(payload)

        states = payload["elements"][0]["states"]
        self.assertIs(states["keyboard_input_mode_switch"], True)
        self.assertEqual("chinese_pinyin", states["current_mode"])
        self.assertEqual("direct_latin", states["target_mode"])

    def test_keyboard_mode_goal_does_not_hide_protocol_extra_compact_element(self) -> None:
        compact = scene_payload()
        compact["elements"] = [
            {
                "element_id": "unsafe-extra-field",
                "role": "keyboard_key",
                "meaning": "language_key",
                "label": "英",
                "bounds": [730, 1130, 810, 1210],
                "confidence": 0.99,
                "states": {"goal_relevant": True},
                "evidence": ["键盘底部按键"],
                "raw_coordinate_hint": [730, 1130],
            }
        ]

        with self.assertRaises(VisionAgentError):
            GenericSceneObserver(SequenceProvider([compact])).observe(
                frames=stable_frames(),
                goal_context={"objective": "把当前键盘切换到英文直输模式"},
            )

    def test_visible_keyboard_input_goal_always_uses_independent_structure_audit(self) -> None:
        compact = scene_payload()
        compact["summary"] = "唯一输入框已聚焦且软键盘可见"
        compact["elements"] = [
            {
                "element_id": "compact-input",
                "role": "input",
                "meaning": "target_text_input",
                "label": "",
                "bounds": [120, 420, 880, 540],
                "confidence": 0.99,
                "states": {
                    "goal_relevant": True,
                    "fully_visible": True,
                    "focused": True,
                    "value": "",
                    "keyboard_layout": "qwerty",
                    "keyboard_input_mode": "chinese_pinyin",
                },
                "evidence": ["完整输入边框和光标"],
            }
        ]
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    bounds=[120, 420, 880, 540],
                    text="",
                    placeholder="",
                )
            ],
            keyboard={
                "visible": True,
                "bounds": [0, 560, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "mode_switch": None,
            },
        )
        audit["application_inputs"][0]["visible_editable_cues"] = ["caret"]
        provider = SequenceProvider([compact, audit])

        scene = GenericSceneObserver(provider).observe(
            frames=stable_frames(),
            goal_context={"objective": "让当前唯一输入框显示 agent，不提交"},
        )

        target = scene.unique_trusted_goal_element()
        self.assertIsNotNone(target)
        self.assertEqual("local_audited_input_1", target.element_id)
        self.assertEqual("direct_latin", target.states["keyboard_input_mode"])
        self.assertEqual(2, provider.calls)

    def test_exact_target_ui_label_resolves_model_over_selection(self) -> None:
        payload = scene_payload()
        payload["elements"] = [
            {
                "element_id": "target",
                "role": "button",
                "meaning": "long_press_target_area",
                "label": "长按我 · 不要移动",
                "bounds": [90, 470, 910, 690],
                "confidence": 0.99,
                "states": {"goal_relevant": True},
                "evidence": ["黄色虚线区域"],
            },
            {
                "element_id": "instruction",
                "role": "text",
                "meaning": "action_instruction",
                "label": "动作：长按黄色区域 800 毫秒",
                "bounds": [110, 390, 890, 450],
                "confidence": 0.99,
                "states": {"goal_relevant": True},
                "evidence": ["操作说明文字"],
            },
        ]

        scene = _parse_scene(
            json.dumps(payload, ensure_ascii=False),
            fingerprint="exact-label",
            goal_context={
                "entities": {"target_ui_label": "长按我 · 不要移动"}
            },
            camera_layout_orientation="portrait",
        )

        self.assertEqual("target", scene.unique_trusted_goal_element().element_id)
        self.assertNotIn("fully_visible", scene.elements[0].states)
        self.assertFalse(scene.elements[1].states["goal_relevant"])

    def test_active_subgoal_target_label_resolves_model_false_relevance(self) -> None:
        payload = scene_payload()
        payload["elements"] = [
            {
                "element_id": "target",
                "role": "button",
                "meaning": "select_acceptance_mode",
                "label": "语义点击",
                "bounds": [150, 380, 850, 460],
                "confidence": 1.0,
                "states": {"goal_relevant": False, "fully_visible": True},
                "evidence": ["列表中唯一逐字匹配的按钮"],
            },
            {
                "element_id": "other",
                "role": "button",
                "meaning": "select_acceptance_mode",
                "label": "系统返回",
                "bounds": [150, 470, 850, 550],
                "confidence": 1.0,
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": ["另一个列表按钮"],
            },
        ]
        context = {
            "entities": {
                "target_ui_label": "语义点击",
                "active_subgoal_visual_context": {
                    "subgoal_id": "open_target_page",
                    "objective": "目标入口对应页面可见",
                    "constraints": [],
                    "completion_conditions": ["目标页面可见"],
                    "execution_class": "navigate",
                    "goal_entities": {"target_ui_label": "语义点击"},
                },
            }
        }

        scene = _parse_scene(
            json.dumps(payload, ensure_ascii=False),
            fingerprint="active-subgoal-exact-label",
            goal_context=context,
            camera_layout_orientation="portrait",
        )

        self.assertTrue(scene.get_element("target").states["goal_relevant"])
        self.assertFalse(scene.get_element("other").states["goal_relevant"])

    def test_exact_target_label_never_promotes_invalid_geometry(self) -> None:
        payload = scene_payload()
        payload["elements"] = [
            {
                "element_id": "invalid-selected-tab",
                "role": "tab",
                "meaning": "current_app_tab",
                "label": "示例应用",
                "bounds": [50, 1450, 280, 1550],
                "confidence": 1.0,
                "states": {
                    "goal_relevant": False,
                    "fully_visible": True,
                    "selected": True,
                },
                "evidence": ["底部当前标签"],
            }
        ]

        scene = _parse_scene(
            json.dumps(payload, ensure_ascii=False),
            fingerprint="invalid-exact-label-geometry",
            goal_context={
                "entities": {"target_ui_label": "示例应用"}
            },
            camera_layout_orientation="portrait",
        )

        self.assertEqual(scene.elements, ())

    def test_observation_prompts_expose_only_active_subgoal_not_future_workflow(self) -> None:
        context = {
            "app_id": "browser",
            "app_name": "浏览器",
            "objective": "先返回桌面，打开浏览器，读取标题，最后返回桌面",
            "entities": {
                "original_goal_visual_context": "打开后读取标题并返回桌面",
                "active_subgoal_visual_context": {
                    "subgoal_id": "open_browser",
                    "objective": "打开浏览器",
                    "constraints": ["仅导航"],
                    "completion_conditions": ["浏览器主界面可见"],
                    "execution_class": "navigate",
                    "goal_entities": {"target_surface": "device"},
                },
            },
        }

        focused = _observation_goal_context(context)
        compact = _compact_prompt(context)
        targeted = _targeted_prompt(context, first_scene=scene_payload())

        self.assertEqual("open_browser", focused["subgoal_id"])
        self.assertEqual("打开浏览器", focused["objective"])
        for prompt in (compact, targeted):
            self.assertIn("打开浏览器", prompt)
            self.assertNotIn("读取标题", prompt)
            self.assertNotIn("返回桌面", prompt)
            self.assertNotIn("original_goal_visual_context", prompt)

    def test_open_app_focus_does_not_refine_for_future_title_goal(self) -> None:
        payload = scene_payload()
        payload.update(
            {
                "foreground_app_id": "browser",
                "screen_id": "browser_home",
                "summary": "当前显示稳定的要闻列表。",
                "elements": [],
                "confidence": 0.98,
            }
        )
        scene = _parse_scene(
            json.dumps(payload, ensure_ascii=False),
            fingerprint="browser-home",
            camera_layout_orientation="portrait",
        )
        context = {
            "app_id": "browser",
            "app_name": "浏览器",
            "objective": "先打开浏览器，随后读取页面标题",
            "entities": {
                "original_goal_visual_context": "打开浏览器后读取页面标题",
                "active_subgoal_visual_context": {
                    "subgoal_id": "open_browser",
                    "objective": "打开浏览器",
                    "constraints": [],
                    "completion_conditions": ["浏览器主界面可见"],
                    "execution_class": "navigate",
                    "goal_entities": {},
                },
            },
        }

        self.assertFalse(_needs_targeted_refinement(scene, context))

        context["entities"]["active_subgoal_visual_context"].update(
            {
                "subgoal_id": "read_page_title",
                "objective": "读取当前页面标题",
                "completion_conditions": ["已读取页面主标题"],
                "execution_class": "observe",
            }
        )
        self.assertTrue(_needs_targeted_refinement(scene, context))

    def test_explicit_system_home_focus_skips_element_target_refinement(self) -> None:
        payload = scene_payload()
        payload.update(
            {
                "foreground_app_id": "news_aggregator",
                "screen_id": "unknown",
                "summary": "当前为新闻流页面，底部有应用内主页标签。",
                "elements": [
                    {
                        "element_id": "app-home-tab",
                        "role": "tab",
                        "meaning": "navigation_tab",
                        "label": "主页",
                        "bounds": [80, 900, 240, 980],
                        "confidence": 0.98,
                        "states": {
                            "goal_relevant": True,
                            "fully_visible": True,
                        },
                        "evidence": ["应用底部导航栏的主页标签"],
                    }
                ],
                "confidence": 0.98,
            }
        )
        contexts = (
            ("返回手机桌面", "手机桌面可见"),
            ("回到手机主屏幕", "手机主屏幕已显示"),
        )
        for index, (objective, condition) in enumerate(contexts):
            with self.subTest(objective=objective):
                provider = SequenceProvider([payload])
                observer = GenericSceneObserver(provider)
                scene = observer.observe(
                    frames=stable_frames(),
                    goal_context={
                        "app_id": "browser",
                        "app_name": "浏览器",
                        "objective": "先返回桌面，然后打开浏览器",
                        "entities": {
                            "active_subgoal_visual_context": {
                                "subgoal_id": f"return_home_{index}",
                                "objective": objective,
                                "constraints": ["仅导航"],
                                "completion_conditions": [condition],
                                "execution_class": "navigate",
                                "goal_entities": {},
                            }
                        },
                    },
                )

                self.assertEqual(1, provider.calls)
                self.assertEqual("unknown", scene.foreground_app_id)
                self.assertEqual("unknown", scene.screen_id)
                self.assertEqual((), scene.elements)
                self.assertFalse(
                    observer.last_diagnostics["targeted_refinement_used"]
                )
                self.assertEqual(
                    "2026-08-19-system-navigation-privacy-view-v1",
                    observer.last_diagnostics[
                        "system_navigation_privacy_view_version"
                    ],
                )

    def test_app_home_and_future_home_do_not_skip_current_target_refinement(self) -> None:
        payload = scene_payload()
        payload.update(
            {
                "foreground_app_id": "unknown",
                "screen_id": "unknown",
                "elements": [],
                "confidence": 0.98,
            }
        )
        scene = _parse_scene(
            json.dumps(payload, ensure_ascii=False),
            fingerprint="unknown-app-page",
            camera_layout_orientation="portrait",
        )
        contexts = (
            {
                "objective": "打开应用主页",
                "entities": {
                    "active_subgoal_visual_context": {
                        "subgoal_id": "open_app_home",
                        "objective": "打开应用主页",
                        "constraints": [],
                        "completion_conditions": ["应用主页可见"],
                        "execution_class": "navigate",
                        "goal_entities": {},
                    }
                },
            },
            {
                "app_id": "browser",
                "app_name": "浏览器",
                "objective": "先打开浏览器，最后返回手机桌面",
                "entities": {
                    "active_subgoal_visual_context": {
                        "subgoal_id": "open_browser",
                        "objective": "打开浏览器",
                        "constraints": [],
                        "completion_conditions": ["浏览器主界面可见"],
                        "execution_class": "navigate",
                        "goal_entities": {},
                    }
                },
            },
        )

        for context in contexts:
            with self.subTest(objective=context["objective"]):
                self.assertTrue(_needs_targeted_refinement(scene, context))

    def test_launcher_app_name_without_trusted_goal_requires_refinement(self) -> None:
        payload = scene_payload()
        payload.update(
            {
                "foreground_app_id": "launcher",
                "screen_id": "home_screen",
                "summary": "当前为稳定桌面，应用图标网格清晰可见。",
                "elements": [
                    {
                        "element_id": "browser-entry",
                        "role": "button",
                        "meaning": "open_browser",
                        "label": "浏览器",
                        "bounds": [150, 40, 310, 150],
                        "confidence": 1.0,
                        "states": {"goal_relevant": False},
                        "evidence": ["蓝色星球图标，下方文字浏览器"],
                    }
                ],
                "confidence": 1.0,
            }
        )
        scene = _parse_scene(
            json.dumps(payload, ensure_ascii=False),
            fingerprint="launcher-browser-entry",
            camera_layout_orientation="portrait",
        )

        for app_id, app_name in (("browser", "浏览器"), ("music", "音乐")):
            with self.subTest(app_id=app_id):
                context = {
                    "app_id": app_id,
                    "app_name": app_name,
                    "objective": f"打开{app_name}",
                    "entities": {
                        "active_subgoal_visual_context": {
                            "subgoal_id": f"open_{app_id}",
                            "objective": f"打开{app_name}",
                            "constraints": [],
                            "completion_conditions": [f"{app_name}主界面可见"],
                            "execution_class": "navigate",
                            "goal_entities": {},
                        }
                    },
                }
                self.assertTrue(_needs_targeted_refinement(scene, context))

    def test_conflicting_root_and_active_target_labels_do_not_rebind(self) -> None:
        payload = scene_payload()
        payload["elements"][0]["label"] = "语义点击"
        payload["elements"][0]["states"] = {"goal_relevant": False}
        context = {
            "entities": {
                "target_ui_label": "系统返回",
                "active_subgoal_visual_context": {
                    "subgoal_id": "open_target_page",
                    "objective": "目标入口对应页面可见",
                    "constraints": [],
                    "completion_conditions": ["目标页面可见"],
                    "execution_class": "navigate",
                    "goal_entities": {"target_ui_label": "语义点击"},
                },
            }
        }

        scene = _parse_scene(
            json.dumps(payload, ensure_ascii=False),
            fingerprint="conflicting-target-labels",
            goal_context=context,
            camera_layout_orientation="portrait",
        )

        self.assertFalse(scene.elements[0].states["goal_relevant"])

    def test_edge_touching_exact_label_does_not_mint_full_visibility(self) -> None:
        payload = scene_payload()
        payload["elements"][0].update(
            {
                "label": "边缘目标",
                "bounds": [0, 600, 260, 760],
                "states": {"goal_relevant": True},
            }
        )

        scene = _parse_scene(
            json.dumps(payload, ensure_ascii=False),
            fingerprint="edge-label",
            goal_context={"entities": {"target_ui_label": "边缘目标"}},
            camera_layout_orientation="portrait",
        )

        self.assertNotIn("fully_visible", scene.elements[0].states)

    def test_duplicate_exact_target_ui_labels_remain_ambiguous(self) -> None:
        payload = scene_payload()
        payload["elements"] = [
            {
                "element_id": f"target-{index}",
                "role": "button",
                "meaning": "drag_target_entry",
                "label": "拖动目标",
                "bounds": [100, 300 + index * 200, 900, 420 + index * 200],
                "confidence": 0.99,
                "states": {"goal_relevant": True},
                "evidence": ["同名目标"],
            }
            for index in range(2)
        ]

        scene = _parse_scene(
            json.dumps(payload, ensure_ascii=False),
            fingerprint="duplicate-label",
            goal_context={"entities": {"target_ui_label": "拖动目标"}},
            camera_layout_orientation="portrait",
        )

        self.assertIsNone(scene.unique_trusted_goal_element())

    def test_camera_alignment_drops_forbidden_peripheral_evidence_only_when_safe_remains(self) -> None:
        payload = scene_payload()
        payload["camera_alignment"]["evidence"] = [
            "顶部PX/MM坐标水平排列",
            "页面文字在手机内容中纵向正立排列",
        ]

        scene = _parse_scene(
            json.dumps(payload, ensure_ascii=False),
            fingerprint="local-fingerprint",
            goal_context={"objective": "读取当前页面"},
            camera_layout_orientation="portrait",
        )

        self.assertEqual(
            ("页面文字在手机内容中纵向正立排列",),
            scene.camera_alignment.evidence,
        )

    def test_camera_alignment_with_only_forbidden_evidence_downgrades_rotation(self) -> None:
        for evidence in (
            ["PX: (264, 151)", "MM: (44.8, 44.4)"],
            ["顶部PX/MM坐标水平排列"],
        ):
            with self.subTest(evidence=evidence):
                payload = scene_payload()
                payload["camera_alignment"]["evidence"] = evidence

                scene = _parse_scene(
                    json.dumps(payload, ensure_ascii=False),
                    fingerprint="local-fingerprint",
                    goal_context={"objective": "读取当前页面"},
                    camera_layout_orientation="portrait",
                )

                self.assertEqual(
                    "portrait", scene.camera_alignment.camera_layout_orientation
                )
                self.assertEqual(
                    "unknown", scene.camera_alignment.phone_content_rotation
                )
                self.assertEqual(0.0, scene.camera_alignment.confidence)
                self.assertEqual((), scene.camera_alignment.evidence)

    def test_camera_alignment_hud_downgrade_does_not_hide_invalid_scalar(self) -> None:
        payload = scene_payload()
        payload["camera_alignment"].update(
            {
                "phone_content_rotation": "unknownish",
                "evidence": ["PX: (264, 151)"],
            }
        )

        with self.assertRaisesRegex(VisionAgentError, "phone_content_rotation"):
            _parse_scene(
                json.dumps(payload, ensure_ascii=False),
                fingerprint="bad-alignment",
                camera_layout_orientation="portrait",
            )

    def test_saved_controller_canvas_shapes_have_distinct_local_orientations(self) -> None:
        self.assertEqual(
            "portrait",
            _camera_layout_orientation(Image.new("RGB", (810, 1440))),
        )
        self.assertEqual(
            "landscape",
            _camera_layout_orientation(Image.new("RGB", (1440, 810))),
        )

    def test_system_ui_goal_uses_independent_three_orientation_audit(self) -> None:
        compact = scene_payload()
        compact["elements"] = []
        compact["confidence"] = 0.65
        compact["system_ui"] = {
            "immersive_or_fullscreen": "likely_true",
            "navigation_bar_visible": "hidden",
        }
        provider = SequenceProvider([compact, system_ui_audit_payload()])
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={"objective": "恢复当前手机的系统导航栏可见状态"},
        )

        self.assertIs(scene.system_ui.immersive_or_fullscreen, True)
        self.assertIs(scene.system_ui.navigation_bar_visible, False)
        self.assertEqual(0.95, scene.confidence)
        self.assertEqual(2, provider.calls)
        self.assertEqual([2600, 600], provider.max_tokens_seen)
        self.assertTrue(observer.last_diagnostics["system_ui_audit_used"])
        self.assertFalse(observer.last_diagnostics["system_ui_audit_retry_used"])
        self.assertFalse(observer.last_diagnostics["targeted_refinement_used"])
        audit_content = provider.messages_seen[1][1]["content"]
        self.assertEqual(
            3,
            sum(item.get("type") == "image_url" for item in audit_content),
        )

    def test_system_ui_audit_fails_closed_without_remote_retry_on_unknown(self) -> None:
        compact = scene_payload()
        compact["elements"] = []
        provider = SequenceProvider(
            [
                compact,
                system_ui_audit_payload(confidence=0.4),
                system_ui_audit_payload(
                    immersive_or_fullscreen="unknown",
                    navigation_bar_visible="unknown",
                ),
            ]
        )
        observer = GenericSceneObserver(provider)

        with self.assertRaisesRegex(VisionAgentError, "系统界面只读审计"):
            observer.observe(
                frames=stable_frames(),
                goal_context={"objective": "检查全屏状态和系统导航栏"},
            )

        self.assertEqual(2, provider.calls)
        self.assertTrue(observer.last_diagnostics["system_ui_audit_used"])
        self.assertFalse(observer.last_diagnostics["system_ui_audit_retry_used"])

    def test_system_ui_audit_rejects_action_fields_and_coordinate_evidence(self) -> None:
        compact = scene_payload()
        compact["elements"] = []
        bad_action = system_ui_audit_payload()
        bad_action["action"] = "swipe"
        bad_evidence = system_ui_audit_payload(evidence=["点击坐标(500,900)"])
        for invalid in (bad_action, bad_evidence):
            with self.subTest(invalid=invalid):
                provider = SequenceProvider([compact, invalid, invalid])
                with self.assertRaisesRegex(VisionAgentError, "系统界面只读审计"):
                    GenericSceneObserver(provider).observe(
                        frames=stable_frames(),
                        goal_context={"objective": "显示系统导航栏"},
                    )
                self.assertEqual(2, provider.calls)

    def test_non_system_ui_goal_does_not_trigger_system_ui_audit(self) -> None:
        payload = scene_payload()
        payload["elements"][0]["states"]["goal_relevant"] = True
        provider = FakeProvider(payload)
        observer = GenericSceneObserver(provider)

        observer.observe(
            frames=stable_frames(),
            goal_context={"objective": "查看数字七"},
        )

        self.assertEqual(1, provider.calls)
        self.assertFalse(observer.last_diagnostics["system_ui_audit_used"])

    def test_system_ui_prohibition_constraint_does_not_trigger_audit(self) -> None:
        payload = scene_payload()
        payload["elements"][0]["states"]["goal_relevant"] = True
        provider = FakeProvider(payload)
        observer = GenericSceneObserver(provider)

        observer.observe(
            frames=stable_frames(),
            goal_context={
                "objective": "显示本地验收模式选择列表",
                "target_ui_label": "验收模式选择列表",
                "constraints": ["禁止操作全屏、屏幕方向和卖家控制栏"],
                "entities": {
                    "original_goal_visual_context": "显示列表并禁止操作全屏"
                },
            },
        )

        self.assertEqual(1, provider.calls)
        self.assertFalse(observer.last_diagnostics["system_ui_audit_used"])

    def test_non_system_ui_goal_downgrades_invalid_fact_values_to_unknown(self) -> None:
        payload = scene_payload()
        payload["elements"][0]["states"]["goal_relevant"] = True
        payload["system_ui"] = {
            "immersive_or_fullscreen": "not_fullscreen",
            "navigation_bar_visible": "visible",
        }
        provider = FakeProvider(payload)
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={"objective": "查看数字七"},
        )

        self.assertEqual("unknown", scene.system_ui.immersive_or_fullscreen)
        self.assertEqual("unknown", scene.system_ui.navigation_bar_visible)
        self.assertEqual(1, provider.calls)
        self.assertFalse(observer.last_diagnostics["system_ui_audit_used"])

    def test_open_tab_goal_excludes_close_glyph_and_group_from_candidates(self) -> None:
        payload = scene_payload()
        payload["elements"] = [
            {
                "element_id": "target-tab",
                "role": "tab",
                "meaning": "generic_verification_page",
                "label": "通用动作真机验...",
                "bounds": [100, 500, 480, 570],
                "confidence": 0.95,
                "states": {"goal_relevant": True},
                "evidence": ["标题可见"],
            },
            {
                "element_id": "close-target-tab",
                "role": "button",
                "meaning": "close_tab",
                "label": "×",
                "bounds": [430, 510, 465, 550],
                "confidence": 0.95,
                "states": {"goal_relevant": True},
                "evidence": ["标签页关闭图标"],
            },
            {
                "element_id": "tab-group",
                "role": "container",
                "meaning": "tab_group",
                "label": "标签页组",
                "bounds": [80, 200, 920, 820],
                "confidence": 0.9,
                "states": {"goal_relevant": True},
                "evidence": ["卡片组"],
            },
        ]

        scene = GenericSceneObserver(FakeProvider(payload)).observe(
            frames=stable_frames(),
            goal_context={"objective": "打开现有的通用动作真机验收标签页"},
        )

        relevant = [
            item.element_id
            for item in scene.elements
            if item.states.get("goal_relevant") is True
        ]
        self.assertEqual(["target-tab"], relevant)

    def test_system_ui_goal_only_fails_closed_invalid_fact_values(self) -> None:
        payload = scene_payload()
        payload["system_ui"] = {
            "immersive_or_fullscreen": "true",
            "navigation_bar_visible": "false",
        }

        scene = _parse_scene(
            json.dumps(payload),
            fingerprint="system-ui-fallback",
            allow_invalid_system_ui_unknown=True,
        )

        self.assertEqual("unknown", scene.system_ui.immersive_or_fullscreen)
        self.assertEqual("unknown", scene.system_ui.navigation_bar_visible)

        payload["system_ui"]["extra"] = False
        with self.assertRaisesRegex(VisionAgentError, "协议外字段"):
            _parse_scene(
                json.dumps(payload),
                fingerprint="system-ui-extra",
                allow_invalid_system_ui_unknown=True,
            )

    def test_scene_parser_requires_explicit_structured_system_ui(self) -> None:
        payload = scene_payload()
        payload.pop("system_ui")

        with self.assertRaisesRegex(VisionAgentError, "必须显式返回 scene.system_ui"):
            _parse_scene(json.dumps(payload), fingerprint="missing-system-ui")

    def test_summary_cannot_override_unknown_system_ui(self) -> None:
        payload = scene_payload()
        payload["summary"] = "系统导航栏清晰可见"
        payload["system_ui"] = {
            "immersive_or_fullscreen": "unknown",
            "navigation_bar_visible": "unknown",
        }

        scene = _parse_scene(json.dumps(payload), fingerprint="unknown-system-ui")

        self.assertEqual("unknown", scene.system_ui.immersive_or_fullscreen)
        self.assertEqual("unknown", scene.system_ui.navigation_bar_visible)

    def test_navigation_bar_fact_cannot_enter_scene_elements(self) -> None:
        payload = scene_payload()
        payload["elements"] = [
            {
                "element_id": "system-bar",
                "role": "container",
                "meaning": "system_nav_bar_stub",
                "label": "系统导航栏区域",
                "bounds": [0, 970, 1000, 1000],
                "confidence": 0.9,
                "states": {"goal_relevant": True},
                "evidence": ["底部导航栏轮廓"],
            }
        ]

        with self.assertRaisesRegex(VisionAgentError, "只能写入 scene.system_ui"):
            _parse_scene(json.dumps(payload, ensure_ascii=False), fingerprint="bar-element")

    def test_observation_prompt_requires_system_ui_without_elements(self) -> None:
        provider = FakeProvider(scene_payload())

        GenericSceneObserver(provider).observe(frames=stable_frames())

        prompt = provider.messages[1]["content"][0]["text"]
        self.assertIn('"system_ui"', prompt)
        self.assertIn('"immersive_or_fullscreen":"unknown"', prompt)
        self.assertIn('"navigation_bar_visible":"unknown"', prompt)
        self.assertIn("绝不得写入elements", prompt)
        self.assertIn("四边独立、可单独识别的色块", prompt)
        self.assertIn("移动源", prompt)
        self.assertIn("fully_visible:true/false", prompt)

    def test_compact_observation_uses_one_image_without_direction_audit(self) -> None:
        provider = FakeProvider(scene_payload())

        scene = GenericSceneObserver(provider).observe(frames=stable_frames())

        content = provider.messages[1]["content"]
        self.assertEqual(
            1,
            sum(item.get("type") == "image_url" for item in content),
        )
        self.assertEqual("portrait", scene.camera_alignment.camera_layout_orientation)
        self.assertEqual("upright", scene.camera_alignment.phone_content_rotation)

    def test_each_independent_direction_audit_uses_one_fresh_three_image_call(self):
        provider = FakeProvider(
            {
                "protocol_version": ORIENTATION_AUDIT_PROTOCOL_VERSION,
                "phone_content_rotation": "upright",
                "confidence": 0.95,
                "evidence": ["手机状态文字正向"],
            }
        )
        observer = GenericSceneObserver(provider)
        frames = stable_frames()

        first = observer.audit_camera_alignment(
            frames=frames,
            device_id="device-a",
            scene_fingerprint="scene-a",
        )
        first_content = provider.messages[1]["content"]
        self.assertEqual(
            3, sum(item.get("type") == "image_url" for item in first_content)
        )
        self.assertEqual(
            ["text", "text", "image_url", "text", "image_url", "text", "image_url"],
            [item.get("type") for item in first_content],
        )
        self.assertEqual(1, provider.calls)
        self.assertEqual(3, observer.last_orientation_audit_diagnostics["image_count"])
        accepted_payload = observer.last_orientation_audit_diagnostics[
            "response_payload"
        ]
        self.assertTrue(accepted_payload["protocol_version_match"])
        self.assertTrue(accepted_payload["rotation_valid"])
        self.assertEqual("upright", accepted_payload["phone_content_rotation"])
        self.assertTrue(accepted_payload["confidence_valid"])
        self.assertEqual(0.95, accepted_payload["confidence"])
        prompt = first_content[0]["text"]
        self.assertIn("Classify ONLY Image 1", prompt)
        self.assertIn("never classification targets", prompt)
        labels = [
            item["text"]
            for item in first_content
            if item.get("type") == "text"
        ]
        self.assertEqual(
            [
                "IMAGE 1 - CLASSIFICATION TARGET - ORIGINAL STABLE FRAME",
                "IMAGE 2 - REFERENCE ONLY - IMAGE 1 ROTATED 90 DEGREES",
                "IMAGE 3 - REFERENCE ONLY - IMAGE 1 ROTATED 270 DEGREES",
            ],
            labels[1:],
        )

        second = observer.audit_camera_alignment(
            frames=frames,
            device_id="device-a",
            scene_fingerprint="scene-a",
        )
        self.assertEqual(2, provider.calls)
        self.assertNotEqual(first.credential_id, second.credential_id)
        self.assertEqual(1, observer.last_orientation_audit_diagnostics["model_calls"])
        self.assertEqual(3, observer.last_orientation_audit_diagnostics["image_count"])
        self.assertFalse(observer.last_orientation_audit_diagnostics["cache_hit"])

        observer.audit_camera_alignment(
            frames=frames,
            device_id="device-a",
            scene_fingerprint="scene-b",
        )
        self.assertEqual(3, provider.calls)

    def test_direction_audit_rejected_evidence_fails_without_remote_retry(self):
        provider = SequenceProvider(
            [
                {
                    "protocol_version": ORIENTATION_AUDIT_PROTOCOL_VERSION,
                    "phone_content_rotation": "upright",
                    "confidence": 0.98,
                    "evidence": ["点击控制端后可让手机保持正向"],
                },
                {
                    "protocol_version": ORIENTATION_AUDIT_PROTOCOL_VERSION,
                    "phone_content_rotation": "upright",
                    "confidence": 0.97,
                    "evidence": ["手机页面文字横向排列且字形正立"],
                },
            ]
        )
        observer = GenericSceneObserver(provider)

        with self.assertRaisesRegex(VisionAgentError, "包含坐标、动作"):
            observer.audit_camera_alignment(
                frames=stable_frames(),
                device_id="device-a",
                scene_fingerprint="scene-a",
            )

        self.assertEqual(1, provider.calls)
        diagnostics = observer.last_orientation_audit_diagnostics
        self.assertFalse(diagnostics["audit_accepted"])
        self.assertFalse(diagnostics["retry_used"])
        self.assertEqual(1, diagnostics["model_calls"])
        self.assertEqual(
            [3],
            [
                sum(
                    item.get("type") == "image_url"
                    for item in messages[1]["content"]
                )
                for messages in provider.messages_seen
            ],
        )
        self.assertEqual(1, len(provider.responses))

    def test_direction_audit_bad_evidence_fails_after_one_call(self):
        invalid = {
            "protocol_version": ORIENTATION_AUDIT_PROTOCOL_VERSION,
            "phone_content_rotation": "upright",
            "confidence": 0.98,
            "evidence": ["点击控制端后可让手机保持正向"],
        }
        provider = SequenceProvider([invalid, invalid, AssertionError("third call")])
        observer = GenericSceneObserver(provider)

        with self.assertRaisesRegex(VisionAgentError, "包含坐标、动作"):
            observer.audit_camera_alignment(
                frames=stable_frames(),
                device_id="device-a",
                scene_fingerprint="scene-a",
            )

        self.assertEqual(1, provider.calls)
        self.assertEqual(2, len(provider.responses))
        diagnostics = observer.last_orientation_audit_diagnostics
        self.assertFalse(diagnostics["audit_accepted"])
        self.assertFalse(diagnostics["retry_used"])
        self.assertEqual(1, diagnostics["model_calls"])

    def test_independent_direction_audit_fails_closed_on_unknown_low_or_extra_fields(self):
        cases = (
            ({"phone_content_rotation": "unknown", "confidence": 0.95}, "未知"),
            ({"phone_content_rotation": "rotated_90", "confidence": 0.95}, "不一致"),
            ({"phone_content_rotation": "upright", "confidence": 0.4}, "置信度"),
            ({"phone_content_rotation": "upright", "confidence": 0.95, "x": 10}, "协议外字段"),
        )
        for mutation, message in cases:
            with self.subTest(mutation=mutation):
                payload = {
                    "protocol_version": ORIENTATION_AUDIT_PROTOCOL_VERSION,
                    "phone_content_rotation": "upright",
                    "confidence": 0.95,
                    "evidence": ["手机状态文字正向"],
                }
                payload.update(mutation)
                provider = FakeProvider(payload)
                observer = GenericSceneObserver(provider)
                with self.assertRaisesRegex(VisionAgentError, message):
                    observer.audit_camera_alignment(
                        frames=stable_frames(),
                        device_id="device-a",
                        scene_fingerprint="scene-a",
                    )
                self.assertEqual(1, provider.calls)
                diagnostics = observer.last_orientation_audit_diagnostics
                self.assertFalse(diagnostics["audit_accepted"])
                self.assertEqual(1, diagnostics["model_calls"])
                self.assertEqual(
                    payload["phone_content_rotation"],
                    diagnostics["response_payload"]["phone_content_rotation"],
                )

    def test_direction_audit_failure_diagnostics_redact_evidence_and_extra_values(self):
        payload = {
            "protocol_version": "private-protocol-value",
            "phone_content_rotation": "sideways-private-value",
            "confidence": "high-private-value",
            "evidence": ["联系人张三 13800138000"],
            "private_instruction": "tap x=123 y=456",
        }
        observer = GenericSceneObserver(FakeProvider(payload))

        with self.assertRaises(VisionAgentError):
            observer.audit_camera_alignment(
                frames=stable_frames(),
                device_id="device-a",
                scene_fingerprint="scene-a",
            )

        diagnostics = observer.last_orientation_audit_diagnostics
        structured = diagnostics["response_payload"]
        serialized = json.dumps(diagnostics, ensure_ascii=False)
        self.assertFalse(structured["protocol_version_match"])
        self.assertFalse(structured["rotation_valid"])
        self.assertNotIn("phone_content_rotation", structured)
        self.assertFalse(structured["confidence_valid"])
        self.assertEqual("string", structured["confidence_type"])
        self.assertNotIn("confidence", structured)
        self.assertTrue(structured["has_unexpected_fields"])
        self.assertEqual(1, structured["unexpected_fields_count"])
        self.assertEqual(1, structured["evidence_count"])
        self.assertEqual(["string"], structured["evidence_item_types"])
        self.assertNotIn("张三", serialized)
        self.assertNotIn("13800138000", serialized)
        self.assertNotIn("sideways-private-value", serialized)
        self.assertNotIn("private-protocol-value", serialized)
        self.assertNotIn("high-private-value", serialized)
        self.assertNotIn("private_instruction", serialized)
        self.assertNotIn("tap x=123", serialized)
        self.assertNotIn("credential", serialized.casefold())
        self.assertFalse(diagnostics["audit_accepted"])
        self.assertEqual(
            diagnostics,
            observer.status()["last_orientation_audit_diagnostics"],
        )

    def test_local_frame_geometry_rejects_model_layout_orientation(self) -> None:
        payload = scene_payload()
        payload["camera_alignment"]["camera_layout_orientation"] = "landscape"

        with self.assertRaisesRegex(VisionAgentError, "本地稳定帧尺寸不一致"):
            _parse_scene(
                json.dumps(payload, ensure_ascii=False),
                fingerprint="layout-mismatch",
                camera_layout_orientation="portrait",
            )

    def test_camera_alignment_requires_strict_non_control_evidence(self) -> None:
        for mutation, error in (
            ({"phone_content_rotation": "unknownish"}, "phone_content_rotation"),
            ({"confidence": "high"}, "confidence"),
            ({"evidence": [123]}, "短字符串"),
        ):
            with self.subTest(mutation=mutation):
                payload = scene_payload()
                payload["camera_alignment"].update(mutation)
                with self.assertRaisesRegex(VisionAgentError, error):
                    _parse_scene(
                        json.dumps(payload, ensure_ascii=False),
                        fingerprint="bad-alignment",
                    )

    def test_scene_parser_requires_explicit_camera_alignment(self) -> None:
        payload = scene_payload()
        payload.pop("camera_alignment")

        with self.assertRaisesRegex(VisionAgentError, "camera_alignment"):
            _parse_scene(json.dumps(payload), fingerprint="missing-alignment")

    def test_visible_keyboard_marks_one_goal_input_focused(self) -> None:
        payload = scene_payload()
        payload["summary"] = "顶部搜索输入框可见，下方显示软键盘"
        payload["elements"] = [
            {
                "element_id": "search-input",
                "role": "input",
                "meaning": "search_query_input",
                "label": "旧文字",
                "bounds": [100, 80, 700, 150],
                "confidence": 0.95,
                "states": {"goal_relevant": True},
                "evidence": ["输入框与键盘同时可见"],
            }
        ]

        scene = _parse_scene(
            json.dumps(payload, ensure_ascii=False),
            fingerprint="frame-focused",
            goal_context={"objective": "修改顶部搜索输入框中的文字"},
        )

        self.assertTrue(scene.elements[0].states["focused"])

    def test_keyboard_does_not_infer_focus_for_multiple_goal_inputs(self) -> None:
        payload = scene_payload()
        payload["summary"] = "表单有两个输入框，下方显示键盘"
        first = {
            "element_id": "input-a",
            "role": "input",
            "meaning": "first_input",
            "label": "A",
            "bounds": [100, 80, 700, 150],
            "confidence": 0.95,
            "states": {"goal_relevant": True},
            "evidence": ["输入框A"],
        }
        second = dict(first)
        second.update(
            {"element_id": "input-b", "meaning": "second_input", "label": "B", "bounds": [100, 180, 700, 250]}
        )
        payload["elements"] = [first, second]

        scene = _parse_scene(
            json.dumps(payload, ensure_ascii=False),
            fingerprint="frame-ambiguous",
            goal_context={"objective": "修改输入框文字"},
        )

        self.assertTrue(all("focused" not in item.states for item in scene.elements))

    def test_unique_input_without_keyboard_is_not_assumed_focused(self) -> None:
        payload = scene_payload()
        payload["summary"] = "顶部搜索输入框可见"
        payload["elements"] = [
            {
                "element_id": "search-input",
                "role": "input",
                "meaning": "search_query_input",
                "label": "旧文字",
                "bounds": [100, 80, 700, 150],
                "confidence": 0.95,
                "states": {"goal_relevant": True},
                "evidence": ["输入框可见"],
            }
        ]

        scene = _parse_scene(
            json.dumps(payload, ensure_ascii=False),
            fingerprint="frame-not-focused",
            goal_context={"objective": "修改顶部搜索输入框中的文字"},
        )

        self.assertNotIn("focused", scene.elements[0].states)

    def test_preserves_keyboard_input_mode_separately_from_qwerty_layout(self) -> None:
        payload = scene_payload()
        payload["summary"] = "顶部空输入框已聚焦，中文拼音 QWERTY 键盘可见"
        payload["elements"] = [
            {
                "element_id": "input-top",
                "role": "input",
                "meaning": "search_input",
                "label": "搜索",
                "bounds": [80, 20, 600, 80],
                "confidence": 0.96,
                "states": {
                    "goal_relevant": True,
                    "focused": True,
                    "value": "",
                    "keyboard_layout": "qwerty",
                    "keyboard_input_mode": "chinese_pinyin",
                },
                "evidence": ["键盘显示中文模式"],
            },
            {
                "element_id": "mode-switch",
                "role": "button",
                "meaning": "switch_keyboard_input_mode",
                "label": "中",
                "bounds": [680, 880, 780, 950],
                "confidence": 0.95,
                "states": {
                    "keyboard_input_mode_switch": True,
                    "current_mode": "chinese_pinyin",
                    "target_mode": "direct_latin",
                },
                "evidence": ["键面显示中"],
            },
        ]

        scene = _parse_scene(
            json.dumps(payload, ensure_ascii=False),
            fingerprint="frame-input-mode",
            goal_context={"objective": "在空输入框输入agent"},
        )

        self.assertEqual("qwerty", scene.elements[0].states["keyboard_layout"])
        self.assertEqual(
            "chinese_pinyin",
            scene.elements[0].states["keyboard_input_mode"],
        )
        self.assertEqual(
            "direct_latin",
            scene.elements[1].states["target_mode"],
        )

    def test_normalizes_casing_for_known_keyboard_enum_tokens(self) -> None:
        payload = scene_payload()
        payload["elements"] = [
            {
                "element_id": "input-top",
                "role": "input",
                "meaning": "search_input",
                "label": "搜索",
                "bounds": [80, 20, 600, 80],
                "confidence": 0.96,
                "states": {
                    "goal_relevant": True,
                    "focused": True,
                    "value": "",
                    "keyboard_layout": " QWERTY ",
                    "keyboard_input_mode": " Chinese_Pinyin ",
                },
                "evidence": ["键盘显示中文模式"],
            },
            {
                "element_id": "mode-switch",
                "role": "button",
                "meaning": "switch_keyboard_input_mode",
                "label": "中",
                "bounds": [680, 880, 780, 950],
                "confidence": 0.95,
                "states": {
                    "keyboard_input_mode_switch": True,
                    "current_mode": " CHINESE_PINYIN ",
                    "target_mode": " DIRECT_LATIN ",
                },
                "evidence": ["键面显示中"],
            },
        ]

        scene = _parse_scene(
            json.dumps(payload, ensure_ascii=False),
            fingerprint="frame-normalized-enums",
            goal_context={"objective": "在空输入框输入agent"},
        )

        self.assertEqual("qwerty", scene.elements[0].states["keyboard_layout"])
        self.assertEqual(
            "chinese_pinyin",
            scene.elements[0].states["keyboard_input_mode"],
        )
        self.assertEqual(
            "direct_latin",
            scene.elements[1].states["target_mode"],
        )

    def test_unknown_keyboard_enum_token_still_fails_closed(self) -> None:
        payload = scene_payload()
        payload["elements"] = [
            {
                "element_id": "input-top",
                "role": "input",
                "meaning": "search_input",
                "label": "搜索",
                "bounds": [80, 20, 600, 80],
                "confidence": 0.96,
                "states": {
                    "goal_relevant": True,
                    "focused": True,
                    "value": "",
                    "keyboard_layout": "alphabetic",
                },
                "evidence": ["键盘布局描述含糊"],
            }
        ]

        with self.assertRaisesRegex(VisionAgentError, "keyboard_layout"):
            _parse_scene(
                json.dumps(payload, ensure_ascii=False),
                fingerprint="frame-invalid-enum",
                goal_context={"objective": "在空输入框输入agent"},
            )

    def test_discards_keyboard_facts_from_non_target_peripheral_container(self) -> None:
        payload = scene_payload()
        payload["elements"] = [
            {
                "element_id": "input-top",
                "role": "input",
                "meaning": "search_input",
                "label": "搜索",
                "bounds": [80, 20, 600, 80],
                "confidence": 0.96,
                "states": {
                    "goal_relevant": True,
                    "focused": True,
                    "value": "",
                    "keyboard_layout": "qwerty",
                    "keyboard_input_mode": "chinese_pinyin",
                },
                "evidence": ["应用输入框与键盘同时可见"],
            },
            {
                "element_id": "kb_layout_qwerty",
                "role": "container",
                "meaning": "keyboard_layout_region",
                "label": "",
                "bounds": [0, 600, 1000, 1000],
                "confidence": 0.95,
                "states": {
                    "goal_relevant": False,
                    "keyboard_layout": "qwerty",
                    "keyboard_input_mode": "chinese_pinyin",
                },
                "evidence": ["非交互键盘区域"],
            },
        ]

        scene = _parse_scene(
            json.dumps(payload, ensure_ascii=False),
            fingerprint="frame-peripheral-keyboard-container",
            goal_context={"objective": "在空输入框输入agent"},
        )

        container = scene.get_element("kb_layout_qwerty")
        self.assertNotIn("keyboard_layout", container.states)
        self.assertNotIn("keyboard_input_mode", container.states)
        self.assertEqual("qwerty", scene.get_element("input-top").states["keyboard_layout"])

    def test_discards_global_keyboard_facts_from_goal_relevant_regular_key(self) -> None:
        payload = scene_payload()
        payload["elements"] = [
            {
                "element_id": "input-top",
                "role": "input",
                "meaning": "search_input",
                "label": "搜索",
                "bounds": [80, 20, 600, 80],
                "confidence": 0.96,
                "states": {
                    "goal_relevant": True,
                    "focused": True,
                    "value": "",
                    "keyboard_layout": "qwerty",
                    "keyboard_input_mode": "direct_latin",
                },
                "evidence": ["唯一输入框与英文键盘同时可见"],
            },
            {
                "element_id": "letter-w",
                "role": "keyboard_key",
                "meaning": "letter_key",
                "label": "w",
                "bounds": [100, 700, 180, 780],
                "confidence": 0.99,
                "states": {
                    "goal_relevant": True,
                    "keyboard_layout": "qwerty",
                    "keyboard_input_mode": "direct_latin",
                },
                "evidence": ["普通字母键 w"],
            },
        ]

        scene = _parse_scene(
            json.dumps(payload, ensure_ascii=False),
            fingerprint="frame-goal-relevant-regular-key",
            goal_context={"objective": "让唯一空白输入框显示 wifi，不提交"},
        )

        key = scene.get_element("letter-w")
        self.assertNotIn("keyboard_layout", key.states)
        self.assertNotIn("keyboard_input_mode", key.states)
        input_states = scene.get_element("input-top").states
        self.assertEqual("qwerty", input_states["keyboard_layout"])
        self.assertEqual("direct_latin", input_states["keyboard_input_mode"])

    def test_goal_relevant_button_cannot_hide_illegal_keyboard_layout(self) -> None:
        payload = scene_payload()
        payload["elements"] = [
            {
                "element_id": "unsafe-button",
                "role": "button",
                "meaning": "unknown_action",
                "label": "继续",
                "bounds": [100, 200, 300, 300],
                "confidence": 0.98,
                "states": {
                    "goal_relevant": True,
                    "keyboard_layout": "qwerty",
                },
                "evidence": ["可动作按钮错误携带全局键盘字段"],
            }
        ]

        with self.assertRaisesRegex(VisionAgentError, "keyboard_layout"):
            _parse_scene(
                json.dumps(payload, ensure_ascii=False),
                fingerprint="frame-target-button-illegal-keyboard-layout",
                goal_context={"objective": "进入下一页面"},
            )

    def test_keyboard_mode_goal_discards_passive_preliminary_container_facts(self) -> None:
        payload = scene_payload()
        payload["elements"] = [
            {
                "element_id": "bad-goal-container",
                "role": "container",
                "meaning": "keyboard_layout_region",
                "label": "",
                "bounds": [0, 600, 1000, 1000],
                "confidence": 0.95,
                "states": {
                    "goal_relevant": True,
                    "keyboard_layout": "qwerty",
                },
                "evidence": ["错误目标结构"],
            }
        ]

        scene = _parse_scene(
            json.dumps(payload, ensure_ascii=False),
            fingerprint="frame-bad-goal-container",
            goal_context={"objective": "切换输入模式"},
        )

        self.assertEqual([], list(scene.elements))

    def test_scene_enum_diagnostics_expose_only_keyboard_tokens(self) -> None:
        payload = scene_payload()
        payload["summary"] = "不应出现在枚举诊断中的页面文字"
        payload["elements"] = [
            {
                "element_id": "input-top",
                "role": "input",
                "meaning": "search_input",
                "label": "敏感输入文字不应出现在诊断中",
                "bounds": [80, 20, 600, 80],
                "confidence": 0.96,
                "states": {
                    "goal_relevant": True,
                    "keyboard_layout": "26-key",
                    "keyboard_input_mode": "Pinyin",
                },
                "evidence": ["不应输出"],
            },
            {
                "element_id": "mode-switch",
                "role": "button",
                "meaning": "switch_keyboard_input_mode",
                "label": "中",
                "bounds": [680, 880, 780, 950],
                "confidence": 0.95,
                "states": {
                    "keyboard_input_mode_switch": True,
                    "current_mode": "Pinyin",
                    "target_mode": "English",
                },
                "evidence": ["不应输出"],
            },
        ]

        diagnostics = _scene_enum_values(
            json.dumps(payload, ensure_ascii=False)
        )

        self.assertEqual(["26-key"], diagnostics["keyboard_layout"])
        self.assertEqual(["Pinyin"], diagnostics["keyboard_input_mode"])
        self.assertEqual(["Pinyin"], diagnostics["current_mode"])
        self.assertEqual(["English"], diagnostics["target_mode"])
        self.assertNotIn("敏感输入文字", json.dumps(diagnostics, ensure_ascii=False))

    def test_known_english_chinese_mode_aliases_normalize_exactly(self) -> None:
        payload = scene_payload()
        payload["elements"] = [
            {
                "element_id": "input-top",
                "role": "input",
                "meaning": "application_text_input",
                "label": "",
                "bounds": [150, 440, 850, 530],
                "confidence": 0.98,
                "states": {
                    "goal_relevant": True,
                    "value": "",
                    "focused": True,
                    "keyboard_layout": "qwerty",
                    "keyboard_input_mode": "English",
                },
                "evidence": ["空输入框和英文键盘可见"],
            },
            {
                "element_id": "mode-switch",
                "role": "button",
                "meaning": "switch_keyboard_input_mode",
                "label": "英",
                "bounds": [720, 880, 800, 940],
                "confidence": 0.97,
                "states": {
                    "goal_relevant": False,
                    "keyboard_input_mode_switch": True,
                    "current_mode": "English",
                    "target_mode": "Chinese",
                },
                "evidence": ["键面显示英"],
            },
        ]

        scene = _parse_scene(
            json.dumps(payload, ensure_ascii=False),
            fingerprint="known-mode-aliases",
            goal_context={"objective": "让当前空白输入框显示 agent"},
        )

        self.assertEqual(
            "direct_latin",
            scene.get_element("input-top").states["keyboard_input_mode"],
        )
        switch = scene.get_element("mode-switch")
        self.assertEqual("direct_latin", switch.states["current_mode"])
        self.assertEqual("chinese_pinyin", switch.states["target_mode"])

    def test_unknown_keyboard_mode_alias_still_fails_closed(self) -> None:
        payload = scene_payload()
        payload["elements"] = [
            {
                "element_id": "input-top",
                "role": "input",
                "meaning": "application_text_input",
                "label": "",
                "bounds": [150, 440, 850, 530],
                "confidence": 0.98,
                "states": {
                    "goal_relevant": True,
                    "value": "",
                    "focused": True,
                    "keyboard_layout": "qwerty",
                    "keyboard_input_mode": "English_US",
                },
                "evidence": ["空输入框可见"],
            }
        ]

        with self.assertRaisesRegex(VisionAgentError, "keyboard_input_mode"):
            _parse_scene(
                json.dumps(payload, ensure_ascii=False),
                fingerprint="unknown-mode-alias",
                goal_context={"objective": "让当前空白输入框显示 agent"},
            )

    def test_clear_goal_binds_unique_nonempty_input_before_focus_inference(self) -> None:
        payload = scene_payload()
        payload["summary"] = '输入框含文字"yi"，右侧有清空图标；软键盘可见'
        payload["elements"] = [
            {
                "element_id": "input_0",
                "role": "input",
                "meaning": "local_search_input",
                "label": "搜索输入框",
                "bounds": [85, 575, 810, 635],
                "confidence": 0.96,
                "states": {"value": "yi", "keyboard_layout": "qwerty"},
                "evidence": ["输入框内文字yi"],
            },
            {
                "element_id": "clear_0",
                "role": "icon",
                "meaning": "clear_local_text",
                "label": "×",
                "bounds": [820, 575, 890, 625],
                "confidence": 0.95,
                "states": {"local_text_clear": True},
                "evidence": ["输入框右侧独立圆形叉号"],
            },
        ]

        scene = _parse_scene(
            json.dumps(payload, ensure_ascii=False),
            fingerprint="frame-clear-bound",
            goal_context={"objective": "把当前输入框中的文字清空"},
        )

        input_element, clear_control = scene.elements
        self.assertTrue(input_element.states["goal_relevant"])
        self.assertTrue(input_element.states["focused"])
        self.assertEqual("yi", input_element.states["value"])
        self.assertTrue(clear_control.states["goal_relevant"])

    def test_clear_goal_does_not_bind_ambiguous_or_distant_structure(self) -> None:
        base_input = {
            "element_id": "input_0",
            "role": "input",
            "meaning": "local_search_input",
            "label": "搜索输入框",
            "bounds": [85, 575, 810, 635],
            "confidence": 0.96,
            "states": {"value": "yi", "keyboard_layout": "qwerty"},
            "evidence": ["输入框内文字yi"],
        }
        clear_control = {
            "element_id": "clear_0",
            "role": "icon",
            "meaning": "clear_local_text",
            "label": "×",
            "bounds": [820, 200, 890, 250],
            "confidence": 0.95,
            "states": {"local_text_clear": True},
            "evidence": ["远离输入框的叉号"],
        }
        payload = scene_payload()
        payload["summary"] = "软键盘可见"
        payload["elements"] = [base_input, clear_control]

        distant_scene = _parse_scene(
            json.dumps(payload, ensure_ascii=False),
            fingerprint="frame-clear-distant",
            goal_context={"objective": "清空当前输入框"},
        )
        self.assertFalse(distant_scene.elements[0].states["goal_relevant"])
        self.assertNotIn("focused", distant_scene.elements[0].states)

        second_input = dict(base_input)
        second_input.update(
            {
                "element_id": "input_1",
                "bounds": [85, 675, 810, 735],
                "states": {"value": "other", "keyboard_layout": "qwerty"},
            }
        )
        bound_clear = dict(clear_control)
        bound_clear["bounds"] = [820, 575, 890, 625]
        payload["elements"] = [base_input, second_input, bound_clear]
        ambiguous_scene = _parse_scene(
            json.dumps(payload, ensure_ascii=False),
            fingerprint="frame-clear-ambiguous",
            goal_context={"objective": "清空当前输入框"},
        )
        self.assertTrue(
            all(item.states.get("goal_relevant") is False for item in ambiguous_scene.elements)
        )
        self.assertTrue(all("focused" not in item.states for item in ambiguous_scene.elements))

    def test_clear_goal_demotes_cancel_text_even_when_model_claims_clear(self) -> None:
        payload = scene_payload()
        payload["summary"] = "搜索输入框含文字agent.com，右侧有取消，软键盘可见"
        payload["elements"] = [
            {
                "element_id": "input_0",
                "role": "input",
                "meaning": "local_search_input",
                "label": "搜索输入框",
                "bounds": [150, 12, 680, 52],
                "confidence": 0.95,
                "states": {
                    "value": "agent.com",
                    "keyboard_layout": "qwerty",
                    "goal_relevant": True,
                    "focused": True,
                },
                "evidence": ["agent.com"],
            },
            {
                "element_id": "cancel_0",
                "role": "button",
                "meaning": "clear_local_text",
                "label": "清除按钮",
                "bounds": [770, 14, 880, 48],
                "confidence": 0.92,
                "states": {"local_text_clear": True, "goal_relevant": True},
                "evidence": ["右侧‘取消’按钮"],
            },
        ]

        scene = _parse_scene(
            json.dumps(payload, ensure_ascii=False),
            fingerprint="frame-cancel-mislabel",
            goal_context={"objective": "清空当前输入框"},
        )

        input_element, cancel = scene.elements
        self.assertFalse(input_element.states["goal_relevant"])
        self.assertNotIn("focused", input_element.states)
        self.assertFalse(cancel.states["goal_relevant"])
        self.assertNotIn("local_text_clear", cancel.states)

    def test_observes_arbitrary_app_and_normalizes_bounds(self) -> None:
        provider = FakeProvider(scene_payload())
        observer = GenericSceneObserver(provider)
        scene = observer.observe(
            frames=stable_frames(),
            goal_context={"objective": "在计算器输入7"},
        )
        self.assertEqual(scene.app_id, "calculator")
        self.assertEqual(scene.foreground_app_id, "calculator")
        self.assertEqual(scene.elements[0].bounds, (0.1, 0.6, 0.26, 0.76))
        self.assertNotEqual(scene.fingerprint, "model-value-must-not-be-trusted")
        self.assertEqual(provider.calls, 1)
        self.assertEqual(provider.max_tokens, 2600)
        self.assertEqual(provider.call_options["timeout"], 60.0)

    def test_low_confidence_scene_is_rejected_before_trusted_observation(self) -> None:
        payload = scene_payload()
        payload["confidence"] = 0.6
        provider = FakeProvider(payload)

        with self.assertRaisesRegex(VisionAgentError, "整体置信度不足"):
            GenericSceneObserver(provider).observe(
                frames=stable_frames(),
                goal_context={"objective": "目标内容可见"},
            )

        self.assertEqual(2, provider.calls)
        self.assertEqual(provider.call_options["max_attempts"], 1)

    def test_low_scene_confidence_accepts_one_strong_goal_element_only(self) -> None:
        payload = scene_payload()
        payload["confidence"] = 0.6
        payload["elements"][0]["states"]["goal_relevant"] = True
        provider = FakeProvider(payload)
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={"objective": "点击唯一清晰目标"},
        )

        self.assertEqual("e1", scene.unique_trusted_goal_element().element_id)
        self.assertEqual("unique_goal_element", observer.last_diagnostics["confidence_basis"])

    def test_low_scene_confidence_rejects_multiple_strong_goal_elements(self) -> None:
        payload = scene_payload()
        payload["confidence"] = 0.6
        payload["elements"][0]["states"]["goal_relevant"] = True
        second = dict(payload["elements"][0])
        second.update({"element_id": "e2", "bounds": [300, 600, 460, 760]})
        payload["elements"].append(second)

        with self.assertRaisesRegex(VisionAgentError, "整体置信度不足"):
            GenericSceneObserver(FakeProvider(payload)).observe(
                frames=stable_frames(),
                goal_context={"objective": "点击目标"},
            )

    def test_low_scene_confidence_rejects_low_confidence_goal_element(self) -> None:
        payload = scene_payload()
        payload["confidence"] = 0.6
        payload["elements"][0]["confidence"] = 0.7
        payload["elements"][0]["states"]["goal_relevant"] = True

        with self.assertRaisesRegex(VisionAgentError, "整体置信度不足"):
            GenericSceneObserver(FakeProvider(payload)).observe(
                frames=stable_frames(),
                goal_context={"objective": "点击目标"},
            )

    def test_low_scene_confidence_accepts_observe_completion_evidence(self) -> None:
        payload = scene_payload()
        payload["confidence"] = 0.6
        payload["elements"][0].update(
            {
                "role": "container",
                "meaning": "visible_result_count",
                "states": {"goal_relevant": True, "fully_visible": True},
            }
        )
        observer = GenericSceneObserver(FakeProvider(payload))

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={"objective": "确认结果已显示"},
        )

        self.assertEqual(("e1",), tuple(x.element_id for x in scene.trusted_completion_evidence()))
        self.assertEqual("completion_evidence_only", observer.last_diagnostics["confidence_basis"])

    def test_unstable_frames_do_not_call_model(self) -> None:
        provider = FakeProvider(scene_payload())
        frames = stable_frames()
        frames[-1] = Image.new("RGB", (540, 960), (255, 255, 255))
        with self.assertRaisesRegex(VisionAgentError, "稳定性检查未通过"):
            GenericSceneObserver(provider).observe(frames=frames)
        self.assertEqual(provider.calls, 0)

    def test_observe_observation_accepts_one_stale_leading_frame_after_convergence(self) -> None:
        provider = FakeProvider(scene_payload())
        settled = Image.new("RGB", (540, 960), (30, 40, 50))
        stale = Image.new("RGB", settled.size, (255, 255, 255))

        observer = GenericSceneObserver(provider)
        observer.observe(
            frames=[stale, settled.copy(), settled.copy(), settled.copy()]
        )

        self.assertEqual(1, provider.calls)
        self.assertTrue(observer.last_diagnostics["local_stability"]["stable"])

    def test_observe_observation_never_reselects_ignored_sharp_leading_frame(self) -> None:
        observer = GenericSceneObserver(FakeProvider(scene_payload()))

        observer.observe(frames=converged_frames_with_sharp_stale_leader())

        diagnostics = observer.last_diagnostics
        self.assertEqual(1, diagnostics["stable_tail_start_index"])
        self.assertNotEqual(0, diagnostics["selected_frame_index"])
        self.assertGreater(
            diagnostics["frame_sharpness_scores"][0],
            max(diagnostics["frame_sharpness_scores"][1:]),
        )

    def test_stable_group_uses_sharpest_frame_instead_of_last_frame(self) -> None:
        observer = GenericSceneObserver(FakeProvider(scene_payload()))
        observer.observe(frames=stable_frames_with_one_sharp_center())
        diagnostics = observer.last_diagnostics
        self.assertEqual(diagnostics["selected_frame_index"], 1)
        scores = diagnostics["frame_sharpness_scores"]
        self.assertEqual(scores[1], max(scores))

    def test_visual_action_field_is_rejected(self) -> None:
        payload = scene_payload()
        payload["action"] = "tap"
        with self.assertRaisesRegex(VisionAgentError, "动作字段"):
            GenericSceneObserver(FakeProvider(payload)).observe(frames=stable_frames())

    def test_goal_context_cannot_smuggle_coordinates(self) -> None:
        with self.assertRaisesRegex(VisionAgentError, "控制字段"):
            GenericSceneObserver(FakeProvider(scene_payload())).observe(
                frames=stable_frames(),
                goal_context={"objective": "打开设置", "x": 50},
            )

    def test_protocol_external_element_field_is_rejected(self) -> None:
        payload = scene_payload()
        payload["elements"][0]["next_action"] = "tap"
        with self.assertRaisesRegex(VisionAgentError, "动作字段"):
            GenericSceneObserver(FakeProvider(payload)).observe(frames=stable_frames())

    def test_home_screen_forces_launcher_even_when_goal_leaks_into_model(self) -> None:
        payload = scene_payload()
        payload["foreground_app_id"] = "douyin"
        payload["screen_id"] = "android_home"
        scene = GenericSceneObserver(FakeProvider(payload)).observe(
            frames=stable_frames(),
            goal_context={"app_id": "douyin", "objective": "打开抖音"},
        )
        self.assertEqual(scene.foreground_app_id, "launcher")

    def test_old_app_id_field_is_accepted_as_compatibility_input(self) -> None:
        payload = scene_payload()
        payload["app_id"] = payload.pop("foreground_app_id")
        scene = GenericSceneObserver(FakeProvider(payload)).observe(
            frames=stable_frames()
        )
        self.assertEqual(scene.foreground_app_id, "calculator")

    def test_irreparable_json_stops_after_one_compact_call(self) -> None:
        provider = SequenceProvider(["{", scene_payload()])
        observer = GenericSceneObserver(provider)
        with self.assertRaisesRegex(VisionAgentError, "唯一、严格有效"):
            observer.observe(frames=stable_frames())
        self.assertEqual(provider.calls, 1)
        self.assertEqual(provider.max_tokens_seen, [2600])
        self.assertEqual(len(provider.responses), 1)
        self.assertFalse(observer.last_diagnostics["compact_retry_used"])
        self.assertEqual(observer.last_diagnostics["model_calls"], 1)
        self.assertEqual(
            len(observer.last_diagnostics["model_call_elapsed_seconds"]),
            1,
        )
        self.assertGreaterEqual(observer.last_diagnostics["elapsed_seconds"], 0.0)

    def test_retry_cannot_rewrite_bounds_object_to_array(self) -> None:
        invalid = scene_payload()
        invalid["elements"][0]["bounds"] = {
            "x": 100,
            "y": 600,
            "width": 160,
            "height": 160,
        }
        provider = SequenceProvider([invalid, scene_payload()])
        with self.assertRaisesRegex(VisionAgentError, "bounds 必须包含4个数值"):
            GenericSceneObserver(provider).observe(frames=stable_frames())
        self.assertEqual(provider.calls, 1)

    def test_retry_cannot_move_overlay_candidate_into_elements(self) -> None:
        invalid = scene_payload()
        candidate = dict(invalid["elements"][0])
        candidate["element_id"] = "add-new"
        candidate["meaning"] = "add_new"
        candidate["label"] = "+"
        candidate["states"] = {"goal_relevant": True}
        invalid["elements"] = []
        invalid["overlays"] = [
            {
                "overlay_id": "add-new",
                "role": "button",
                "meaning": "add_new",
                "bounds": [100, 600, 260, 760],
                "confidence": 0.98,
            }
        ]
        repaired = scene_payload()
        repaired["elements"] = [candidate]
        repaired["overlays"] = ["window_manager"]
        provider = SequenceProvider([invalid, repaired])

        observer = GenericSceneObserver(provider)
        with self.assertRaisesRegex(VisionAgentError, "overlays"):
            observer.observe(
                frames=stable_frames(),
                goal_context={"objective": "让新的空白页面可见"},
            )

        self.assertEqual(1, provider.calls)

    def test_exact_extra_brace_response_is_repaired_after_one_model_call(self) -> None:
        malformed = extra_brace_scene_response()
        self.assertEqual(len(malformed), 608)
        self.assertIn('"evidence":["浏览器"]}}],"overlays"', malformed)
        provider = SequenceProvider([malformed])
        observer = GenericSceneObserver(provider)

        scene = observer.observe(frames=stable_frames())

        self.assertEqual(scene.foreground_app_id, "launcher")
        self.assertEqual(scene.elements[0].label, "浏览器")
        self.assertEqual(provider.calls, 1)
        self.assertEqual(provider.max_tokens_seen, [2600])
        self.assertFalse(observer.last_diagnostics["compact_retry_used"])
        self.assertTrue(observer.last_diagnostics["local_structural_repair_used"])
        self.assertEqual(observer.last_diagnostics["model_calls"], 1)
        self.assertFalse(observer.status()["hardware_actions_enabled"])

    def test_exact_extra_brace_response_has_one_local_strict_candidate(self) -> None:
        scene = _parse_scene_after_unique_structural_edit(
            extra_brace_scene_response(),
            fingerprint="stable-fingerprint",
            camera_layout_orientation="portrait",
        )
        self.assertIsNotNone(scene)
        assert scene is not None
        self.assertEqual(scene.elements[0].label, "浏览器")

    def test_missing_final_brace_is_repaired_only_as_one_strict_scene(self) -> None:
        valid = json.dumps(
            scene_payload(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        provider = SequenceProvider([valid[:-1]])
        observer = GenericSceneObserver(provider)

        scene = observer.observe(frames=stable_frames())

        self.assertEqual(scene.foreground_app_id, "calculator")
        self.assertEqual(provider.calls, 1)
        self.assertEqual(provider.max_tokens_seen, [2600])
        self.assertTrue(observer.last_diagnostics["local_structural_repair_used"])

    def test_missing_two_final_braces_still_fails_closed(self) -> None:
        valid = json.dumps(
            scene_payload(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        provider = SequenceProvider([valid[:-2]])
        observer = GenericSceneObserver(provider)

        with self.assertRaisesRegex(VisionAgentError, "唯一、严格有效"):
            observer.observe(frames=stable_frames())

        self.assertEqual(provider.calls, 1)
        self.assertEqual(provider.max_tokens_seen, [2600])
        self.assertFalse(observer.last_diagnostics["local_structural_repair_used"])

    def test_valid_duplicate_key_first_response_fails_closed(self) -> None:
        valid = json.dumps(scene_payload(), ensure_ascii=False, separators=(",", ":"))
        duplicate = valid.replace(
            '"summary":',
            '"summary":"duplicate","summary":',
            1,
        )
        provider = SequenceProvider([duplicate, valid])
        observer = GenericSceneObserver(provider)

        with self.assertRaisesRegex(VisionAgentError, "重复 JSON 字段"):
            observer.observe(frames=stable_frames())

        self.assertEqual(provider.calls, 1)
        self.assertFalse(observer.last_diagnostics["repair_retry_success"])
        self.assertFalse(observer.status()["hardware_actions_enabled"])

    def test_structural_edit_must_still_pass_strict_scene_validation(self) -> None:
        payload = scene_payload()
        payload["unexpected"] = "forbidden"
        valid = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        malformed = valid.replace('}],"overlays"', '}}],"overlays"', 1)
        provider = SequenceProvider([malformed, scene_payload()])
        observer = GenericSceneObserver(provider)

        with self.assertRaises(VisionAgentError):
            observer.observe(frames=stable_frames())

        self.assertEqual(provider.calls, 1)
        self.assertEqual(provider.max_tokens_seen, [2600])
        self.assertEqual(len(provider.responses), 1)
        self.assertFalse(observer.last_diagnostics["repair_retry_success"])
        self.assertEqual(observer.last_diagnostics["model_calls"], 1)
        self.assertFalse(observer.status()["hardware_actions_enabled"])

    def test_multiple_strict_structural_edit_candidates_fail_closed(self) -> None:
        first = json.dumps(scene_payload(), ensure_ascii=False)
        second_payload = scene_payload()
        second_payload["elements"][0]["label"] = "8"
        second = json.dumps(second_payload, ensure_ascii=False)

        with patch(
            "generic_scene_observer._single_json_structural_edits",
            return_value=iter([first, second]),
        ):
            scene = _parse_scene_after_unique_structural_edit(
                "{}",
                fingerprint="stable-fingerprint",
                camera_layout_orientation="portrait",
            )

        self.assertIsNone(scene)

    def test_structural_repair_rejects_duplicate_json_keys(self) -> None:
        malformed = extra_brace_scene_response().replace(
            '"summary":',
            '"summary":"duplicate","summary":',
            1,
        )

        scene = _parse_scene_after_unique_structural_edit(
            malformed,
            fingerprint="stable-fingerprint",
            camera_layout_orientation="portrait",
        )

        self.assertIsNone(scene)

    def test_structural_repair_has_hard_size_and_candidate_limits(self) -> None:
        oversized = '{"summary":"' + (
            "x" * _MAX_JSON_STRUCTURAL_REPAIR_CHARS
        ) + '"}}'
        self.assertEqual(list(_single_json_structural_edits(oversized)), [])

        payload = scene_payload()
        payload["summary"] = "x" * 14000
        valid = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        malformed = valid.replace('}],"overlays"', '}}],"overlays"', 1)
        candidates = list(_single_json_structural_edits(malformed))
        self.assertGreater(len(candidates), 0)
        self.assertLessEqual(
            len(candidates),
            _MAX_JSON_STRUCTURAL_REPAIR_CANDIDATES,
        )

    def test_service_disconnect_is_not_misclassified_as_format_retry(self) -> None:
        provider = SequenceProvider(
            [VisionAgentError("千问视觉连接连续1次中断：Server disconnected")]
        )
        observer = GenericSceneObserver(provider)
        with self.assertRaisesRegex(VisionAgentError, "Server disconnected"):
            observer.observe(frames=stable_frames())
        self.assertEqual(provider.calls, 1)
        self.assertFalse(observer.last_diagnostics["format_retry_used"])
        self.assertEqual(observer.last_diagnostics["error_type"], "service_disconnect")
        self.assertEqual(observer.last_diagnostics["model_calls"], 1)
        self.assertIn("未建立可信候选", observer.last_diagnostics["safe_stop_reason"])

    def test_targeted_single_punctuation_error_uses_local_repair(self) -> None:
        first = scene_payload()
        first["elements"] = []
        first["summary"] = "未知首页"
        provider = SequenceProvider([first, extra_brace_targeted_delta_response()])
        observer = GenericSceneObserver(provider)
        scene = observer.observe(
            frames=stable_frames(),
            goal_context={"objective": "查找目标按钮"},
        )
        self.assertEqual(scene.elements[0].element_id, "e1")
        self.assertEqual(provider.calls, 2)
        self.assertEqual(provider.max_tokens_seen, [2600, 2600])
        self.assertTrue(observer.last_diagnostics["format_retry_used"])
        self.assertTrue(observer.last_diagnostics["local_structural_repair_used"])
        self.assertTrue(observer.last_diagnostics["repair_retry_success"])

    def test_targeted_unrepairable_json_does_not_sample_again(self) -> None:
        first = scene_payload()
        first["elements"] = []
        first["summary"] = "未知首页"
        provider = SequenceProvider([first, "{", targeted_delta_payload()])
        observer = GenericSceneObserver(provider)

        with self.assertRaisesRegex(VisionAgentError, "不存在唯一、严格有效"):
            observer.observe(
                frames=stable_frames(),
                goal_context={"objective": "查找目标按钮"},
            )

        self.assertEqual(provider.calls, 2)
        self.assertEqual(provider.max_tokens_seen, [2600, 2600])
        self.assertEqual(len(provider.responses), 1)
        self.assertFalse(observer.last_diagnostics["local_structural_repair_used"])
        self.assertFalse(observer.last_diagnostics["repair_retry_success"])

    def test_compact_observation_never_uses_remote_format_repair(self) -> None:
        first_retry = scene_payload()
        first_retry["elements"] = []
        first_retry["summary"] = "未知首页"
        provider = SequenceProvider(["{", first_retry])
        observer = GenericSceneObserver(provider)
        with self.assertRaises(VisionAgentError):
            observer.observe(
                frames=stable_frames(),
                goal_context={"objective": "查找目标按钮"},
            )
        self.assertEqual(provider.calls, 1)
        self.assertEqual(provider.max_tokens_seen, [2600])
        self.assertEqual(len(provider.responses), 1)
        self.assertFalse(observer.last_diagnostics["format_retry_used"])
        self.assertFalse(observer.last_diagnostics["repair_retry_success"])

    def test_single_evidence_string_is_normalized_before_strict_validation(self) -> None:
        payload = scene_payload()
        payload["elements"][0]["evidence"] = "7"
        scene = GenericSceneObserver(FakeProvider(payload)).observe(
            frames=stable_frames()
        )
        self.assertEqual(scene.elements[0].evidence, ("7",))

    def test_evidence_object_is_not_silently_repaired(self) -> None:
        payload = scene_payload()
        payload["elements"][0]["evidence"] = {"text": "7"}
        with self.assertRaisesRegex(VisionAgentError, "evidence 必须是数组"):
            GenericSceneObserver(FakeProvider(payload)).observe(frames=stable_frames())

    def test_unsupported_peripheral_structure_is_discarded(self) -> None:
        payload = scene_payload()
        payload["elements"].insert(
            0,
            {
                "element_id": "tabs",
                "role": "tab_group",
                "meaning": "top_channel_group",
                "label": "频道栏",
                "bounds": [100, 0, 900, 60],
                "confidence": 0.96,
                "states": {},
                "evidence": ["推荐"],
            },
        )
        scene = GenericSceneObserver(FakeProvider(payload)).observe(
            frames=stable_frames()
        )
        self.assertEqual([item.element_id for item in scene.elements], ["e1"])

    def test_goal_relevant_container_is_preserved(self) -> None:
        payload = scene_payload()
        payload["elements"][0]["role"] = "container"
        payload["elements"][0]["meaning"] = "video_content"
        payload["elements"][0]["states"] = {"goal_relevant": True}
        scene = GenericSceneObserver(FakeProvider(payload)).observe(
            frames=stable_frames()
        )
        self.assertEqual(scene.elements[0].role, "container")
        self.assertEqual(scene.elements[0].meaning, "video_content")

    def test_unsupported_goal_target_still_stops_controller(self) -> None:
        payload = scene_payload()
        payload["elements"][0]["role"] = "search_icon"
        payload["elements"][0]["states"] = {"goal_relevant": True}
        with self.assertRaisesRegex(VisionAgentError, "不支持的元素角色"):
            GenericSceneObserver(FakeProvider(payload)).observe(frames=stable_frames())

    def test_missing_goal_element_triggers_targeted_refinement(self) -> None:
        first = scene_payload()
        first["foreground_app_id"] = "launcher"
        first["screen_id"] = "android_home"
        first["summary"] = "安卓桌面"
        first["elements"] = []
        refined = targeted_delta_payload(elements=[
            {
                "element_id": "e1",
                "role": "icon",
                "meaning": "app_icon",
                "label": "微信",
                "bounds": [200, 700, 340, 850],
                "confidence": 0.96,
                "states": {"goal_relevant": True},
                "evidence": ["微信"],
            }
        ])
        provider = SequenceProvider([first, refined])
        observer = GenericSceneObserver(provider)
        scene = observer.observe(
            frames=stable_frames(),
            goal_context={
                "app_id": "wechat",
                "app_name": "微信",
                "objective": "打开微信",
            },
        )
        self.assertEqual(provider.calls, 2)
        self.assertEqual(provider.max_tokens_seen, [2600, 2600])
        self.assertEqual(scene.elements[0].label, "微信")
        self.assertTrue(observer.last_diagnostics["targeted_refinement_used"])
        targeted_text = provider.messages_seen[1][1]["content"][0]["text"]
        self.assertIn("置信度只评价当前画面观察本身是否可靠", targeted_text)
        self.assertIn("系统级动作没有屏内按钮", targeted_text)
        self.assertIn("目标相关控件确实不存在时返回空elements", targeted_text)
        self.assertIn("不能因为目标尚未完成而降低", targeted_text)
        self.assertIn("模糊、遮挡或不唯一时仍必须降低", targeted_text)

    def test_goal_overflow_discards_all_compact_geometry_and_forces_targeted(self) -> None:
        first = scene_payload()
        first["foreground_app_id"] = "wechat"
        first["screen_id"] = "wechat_home"
        first["summary"] = "微信首页聊天列表可见"
        first["elements"] = [
            {
                "element_id": "compact-tab",
                "role": "tab",
                "meaning": "nav_wechat",
                "label": "微信",
                "bounds": [60, 1750, 290, 1880],
                "confidence": 1.0,
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": ["底部微信标签"],
            },
            {
                "element_id": "compact-chat",
                "role": "list_item",
                "meaning": "chat_entry",
                "label": "文件传输助手",
                "bounds": [60, 420, 940, 580],
                "confidence": 1.0,
                "states": {"goal_relevant": False, "fully_visible": True},
                "evidence": ["聊天列表第三项"],
            },
        ]
        refined = targeted_delta_payload(
            elements=[
                {
                    "element_id": "target-chat",
                    "role": "list_item",
                    "meaning": "chat_entry",
                    "label": "文件传输助手",
                    "bounds": [80, 280, 920, 400],
                    "confidence": 1.0,
                    "states": {"goal_relevant": True, "fully_visible": True},
                    "evidence": ["聊天列表中完整可见的文件传输助手"],
                }
            ]
        )
        provider = SequenceProvider([first, refined])
        observer = GenericSceneObserver(provider)
        scene = observer.observe(
            frames=stable_frames(),
            goal_context={
                "app_id": "wechat",
                "app_name": "微信",
                "objective": "打开文件传输助手",
                "entities": {"target_ui_label": "文件传输助手"},
            },
        )
        self.assertEqual(provider.calls, 2)
        self.assertEqual([item.element_id for item in scene.elements], ["target-chat"])
        self.assertTrue(observer.last_diagnostics["compact_geometry_discarded"])
        self.assertTrue(observer.last_diagnostics["targeted_refinement_used"])

    def test_targeted_overflow_after_compact_discard_still_fails_closed(self) -> None:
        first = scene_payload()
        first["elements"][0]["states"] = {"goal_relevant": True}
        first["elements"][0]["bounds"] = [60, 1750, 290, 1880]
        refined = targeted_delta_payload(elements=[dict(first["elements"][0])])
        provider = SequenceProvider([first, refined])
        observer = GenericSceneObserver(provider)
        with self.assertRaisesRegex(
            VisionAgentError,
            "bounds 超出归一化画面",
        ):
            observer.observe(
                frames=stable_frames(),
                goal_context={"objective": "打开目标页面"},
            )
        self.assertEqual(provider.calls, 2)
        self.assertTrue(observer.last_diagnostics["compact_geometry_discarded"])

    def test_targeted_literal_target_discards_only_future_invalid_input_geometry(self) -> None:
        base_payload = scene_payload()
        base_payload["foreground_app_id"] = "wechat"
        base_payload["screen_id"] = "unknown"
        base_payload["summary"] = "微信对话页"
        base_payload["elements"] = []
        context = {
            "objective": "从当前页面进入文件传输助手后保留未发送草稿",
            "entities": {
                "target_ui_label": "文件传输助手",
                "input_text": "codex",
                "active_subgoal_visual_context": {
                    "subgoal_id": "navigate_to_file_transfer",
                    "objective": "文件传输助手页面在前台可见",
                    "constraints": ["不得选择其他联系人"],
                    "completion_conditions": ["文件传输助手页面在前台可见"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "target_ui_label": "文件传输助手",
                        "input_text": "codex",
                    },
                },
            },
        }
        targeted = targeted_delta_payload(
            elements=[
                {
                    "element_id": "page-title",
                    "role": "text",
                    "meaning": "page_title",
                    "label": "文件传输助手",
                    "bounds": [340, 15, 660, 55],
                    "confidence": 1.0,
                    "states": {"fully_visible": True},
                    "evidence": ["页面顶部中央清晰显示文件传输助手"],
                },
                {
                    "element_id": "future-input",
                    "role": "input",
                    "meaning": "message_text_field",
                    "label": "",
                    "bounds": [180, 1670, 680, 1790],
                    "confidence": 1.0,
                    "states": {"fully_visible": True, "value": ""},
                    "evidence": ["后续子目标的底部输入框"],
                },
            ]
        )

        base = _parse_scene(
            json.dumps(base_payload, ensure_ascii=False),
            fingerprint="local-fingerprint",
            goal_context=context,
            camera_layout_orientation="portrait",
        )
        scene = _parse_targeted_scene_delta(
            json.dumps(targeted, ensure_ascii=False),
            base_scene=base,
            fingerprint="local-fingerprint",
            goal_context=context,
        )

        self.assertEqual(["page-title"], [item.element_id for item in scene.elements])
        self.assertEqual(
            "文件传输助手",
            scene.unique_trusted_goal_element().label,
        )

    def test_input_goal_isolates_compact_geometry_for_dedicated_audit(self) -> None:
        first = scene_payload()
        first["elements"][0]["role"] = "input"
        first["elements"][0]["meaning"] = "message_input"
        first["elements"][0]["states"] = {"goal_relevant": True, "focused": True}
        first["elements"][0]["bounds"] = [60, 1750, 940, 1880]
        first["elements"].append(
            {
                "element_id": "passive-keyboard-container",
                "role": "container",
                "meaning": "soft_keyboard_area",
                "label": "QWERTY Keyboard",
                "bounds": [0, 640, 1000, 1000],
                "confidence": 1.0,
                "states": {
                    "goal_relevant": True,
                    "fully_visible": True,
                    "keyboard_layout": "qwerty",
                    "keyboard_input_mode": "direct_latin",
                },
                "evidence": ["底部可见完整字母键盘区域"],
            }
        )
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(text="", placeholder="请输入")
            ]
        )
        provider = SequenceProvider([first, audit])
        observer = GenericSceneObserver(provider)
        scene = observer.observe(
            frames=stable_frames(),
            goal_context={
                "objective": "输入框内容为 codex 且不提交",
                "entities": {"input_text": "codex", "spatial_hint": "bottom"},
            },
        )
        self.assertEqual(provider.calls, 2)
        self.assertFalse(observer.last_diagnostics["compact_geometry_discarded"])
        self.assertTrue(observer.last_diagnostics["compact_input_geometry_isolated"])
        self.assertFalse(observer.last_diagnostics["targeted_refinement_used"])
        self.assertEqual(
            "local_audited_input_1",
            scene.unique_trusted_goal_element().element_id,
        )
        self.assertNotIn(
            "passive-keyboard-container",
            {item.element_id for item in scene.elements},
        )

    def test_pending_candidate_ledger_ignores_conflicting_compact_input_value(
        self,
    ) -> None:
        first = scene_payload()
        first["foreground_app_id"] = "unknown"
        first["screen_id"] = "multiline_input_acceptance"
        first["summary"] = "正文仍为空，键盘显示预测栏"
        first["elements"] = [
            {
                "element_id": "compact-input-shadow",
                "role": "input",
                "meaning": "application_text_input",
                "label": "正文",
                "bounds": [140, 270, 860, 450],
                "confidence": 1.0,
                "states": {
                    "goal_relevant": True,
                    "fully_visible": True,
                    "focused": True,
                    "value": "错误粗值",
                },
                "evidence": ["compact 自由转写不得成为正文权威"],
            }
        ]
        anchors = {
            "q": [120, 710],
            "p": [860, 710],
            "a": [160, 790],
            "l": [800, 790],
            "z": [260, 870],
            "m": [720, 870],
            "backspace": [840, 870],
        }
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="body-field",
                    bounds=[140, 270, 860, 450],
                    text="",
                    placeholder="",
                    field_labels=["正文"],
                    visible_editable_cues=["bordered input area"],
                    caret_line_index=0,
                )
            ],
            ime_preedit_regions=[
                {
                    "region_id": "misclassified-full-field-text",
                    "bounds": [140, 270, 860, 450],
                    "text": "你好",
                    "confidence": 1.0,
                    "candidates": [
                        {
                            "text": "你好",
                            "bounds": [110, 590, 230, 630],
                            "confidence": 1.0,
                            "fully_visible": True,
                        },
                        {
                            "text": "好",
                            "bounds": [250, 590, 330, 630],
                            "confidence": 1.0,
                            "fully_visible": True,
                        },
                    ],
                }
            ],
            keyboard={
                "visible": True,
                "bounds": [80, 570, 920, 1000],
                "layout": "qwerty",
                "input_mode": "chinese_pinyin",
                "case_mode": "lower",
                "qwerty_anchors": anchors,
                "mode_switch": None,
                "backspace_key": None,
                "enter_key": {
                    "label": "↵",
                    "bounds": [800, 900, 920, 970],
                    "confidence": 1.0,
                    "fully_visible": True,
                    "key_action": "newline",
                },
            },
        )
        lineage = TypedInputLineage(
            version=TYPED_INPUT_LINEAGE_VERSION,
            device_id="device-local-01",
            exact_value="你好",
            app_id="unknown",
            screen_id="multiline_input_acceptance",
            input_meaning="application_text_input",
            input_field_id="input_field_1",
            input_bounds=(0.14, 0.27, 0.86, 0.45),
            before_fingerprint="before-candidate",
            after_fingerprint="pending-visual-verification",
            action_digest="a" * 64,
            receipt_digest="b" * 64,
            surface_descriptors=(),
            recorded_at_epoch=time.time(),
            source="pending_verified_ime_candidate_action",
        )
        lineage.validate()
        provider = SequenceProvider([first, audit])
        observer = GenericSceneObserver(
            provider,
            qwerty_row_snapper=lambda _frames, _anchors: anchors,
        )

        scene = observer.observe(
            frames=stable_frames(),
            device_id="device-local-01",
            input_lineage_override=lineage,
            goal_context={
                "entities": {
                    "active_subgoal_visual_context": {
                        "subgoal_id": "input_exact_text",
                        "objective": "在正文输入两行中文",
                        "constraints": ["不得发送"],
                        "completion_conditions": [
                            "正文逐字等于你好换行世界"
                        ],
                        "execution_class": "navigate",
                        "goal_entities": {
                            "input_text": "你好\n世界",
                            "active_input_transaction_text": "你好\n世界",
                            "active_input_field_id": "input_field_1",
                            "active_input_field_label": "正文",
                            "active_input_multiline": True,
                        },
                    }
                }
            },
        )

        self.assertEqual(2, provider.calls)
        self.assertTrue(observer.last_diagnostics["input_lineage_used"])
        field = scene.get_element("local_audited_input_1")
        self.assertEqual("你好", field.states["value"])
        self.assertNotIn("ime_preedit_text", field.states)
        enter = scene.get_element("local_audited_enter_key_1")
        self.assertEqual("你好\n", enter.states["expected_input_value"])

    def test_passive_keyboard_container_isolation_keeps_ambiguous_items_strict(self) -> None:
        context = {
            "objective": "输入框内容为 codex 且不提交",
            "entities": {"input_text": "codex"},
        }
        base = {
            "element_id": "candidate",
            "role": "container",
            "meaning": "soft_keyboard_area",
            "label": "QWERTY Keyboard",
            "bounds": [0, 640, 1000, 1000],
            "confidence": 1.0,
            "states": {
                "goal_relevant": True,
                "keyboard_layout": "qwerty",
            },
            "evidence": ["底部字母键盘区域"],
        }
        variants = {
            "action-bearing": {**base, "states": {**base["states"], "plan": "tap"}},
            "input-role": {**base, "role": "input"},
            "non-keyboard": {**base, "meaning": "content_panel"},
            "missing-keyboard-state": {
                **base,
                "states": {"goal_relevant": True, "fully_visible": True},
            },
            "invalid-bounds": {**base, "bounds": [0, 640, 1001, 1000]},
        }
        for name, candidate in variants.items():
            with self.subTest(name=name):
                payload = {"elements": [candidate]}
                self.assertFalse(
                    _strip_preliminary_keyboard_containers_for_dedicated_audit(
                        payload,
                        context,
                    )
                )
                self.assertEqual([candidate], payload["elements"])

    def test_empty_input_audit_retries_once_without_using_first_result(self) -> None:
        first = scene_payload()
        first["elements"][0]["role"] = "input"
        first["elements"][0]["meaning"] = "message_input"
        first["elements"][0]["states"] = {
            "goal_relevant": True,
            "focused": True,
        }
        first["elements"][0]["bounds"] = [60, 1750, 940, 1880]
        empty_audit = input_audit_payload(application_inputs=[])
        recovered_audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="bottom-draft-input",
                    bounds=[150, 786, 680, 905],
                    text="",
                    placeholder="",
                    right_button=None,
                )
            ]
        )
        provider = SequenceProvider([first, empty_audit, recovered_audit])
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={
                "objective": "在底部唯一消息输入框中保留 codex 草稿",
                "entities": {"input_text": "codex", "spatial_hint": "bottom"},
            },
        )

        self.assertEqual(3, provider.calls)

        self.assertEqual(
            "local_audited_input_1",
            scene.unique_trusted_goal_element().element_id,
        )
        self.assertEqual(
            (0.15, 0.91, 0.68, 0.96),
            scene.unique_trusted_goal_element().bounds,
        )
        self.assertTrue(observer.last_diagnostics["input_structure_audit_used"])
        self.assertTrue(
            observer.last_diagnostics["input_structure_audit_retry_used"]
        )
        retry_content = provider.messages_seen[2][1]["content"]
        self.assertEqual(
            1,
            sum(item.get("type") == "image_url" for item in retry_content),
        )
        self.assertIn("crop-local 0..1000", retry_content[0]["text"])
        self.assertNotEqual(
            provider.messages_seen[1][1]["content"],
            retry_content,
        )

    def test_empty_input_audit_uses_unique_compact_field_only_as_crop_hint(
        self,
    ) -> None:
        first = scene_payload()
        first["elements"][0]["role"] = "input"
        first["elements"][0]["meaning"] = "message_input"
        first["elements"][0]["label"] = "消息"
        first["elements"][0]["bounds"] = [120, 430, 880, 520]
        first["elements"][0]["states"] = {
            "goal_relevant": True,
            "fully_visible": True,
            "focused": True,
            "value": "",
        }
        empty_audit = input_audit_payload(application_inputs=[])
        recovered_audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="message-field",
                    bounds=[140, 80, 860, 210],
                    text="",
                    placeholder="消息",
                )
            ]
        )
        provider = SequenceProvider([first, empty_audit, recovered_audit])
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={
                "objective": "在唯一消息输入框输入 cross app text",
                "entities": {
                    "input_text": "cross app text",
                    "active_subgoal_visual_context": {
                        "subgoal_id": "input_text",
                        "objective": "在唯一消息输入框输入 cross app text",
                        "constraints": [],
                        "completion_conditions": [
                            "输入框逐字等于 cross app text"
                        ],
                        "execution_class": "navigate",
                        "goal_entities": {
                            "input_text": "cross app text",
                            "active_input_transaction_text": "cross app text",
                        },
                    },
                },
            },
        )

        self.assertEqual(3, provider.calls)
        self.assertEqual(
            "local_audited_input_1",
            scene.unique_trusted_goal_element().element_id,
        )
        self.assertEqual(
            (0.14, 0.255, 0.86, 0.36),
            scene.unique_trusted_goal_element().bounds,
        )
        self.assertTrue(observer.last_diagnostics["input_structure_audit_retry_used"])

    def test_read_only_input_locator_retries_from_unique_compact_field_hint(
        self,
    ) -> None:
        first = scene_payload()
        first["elements"][0].update(
            {
                "role": "input",
                "meaning": "application_text_input",
                "label": "正文",
                "bounds": [130, 590, 870, 740],
                "states": {
                    "goal_relevant": True,
                    "fully_visible": True,
                    "focused": True,
                    "value": "",
                },
            }
        )
        recovered = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="multiline-field",
                    bounds=[130, 80, 870, 230],
                    text="",
                    placeholder="正文",
                )
            ]
        )
        provider = SequenceProvider(
            [first, input_audit_payload(application_inputs=[]), recovered]
        )
        context = {
            "objective": "输入两行文本",
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "locate_input",
                    "objective": "定位当前页面唯一的多行输入框，并确认其可见",
                    "constraints": ["不得点击或输入"],
                    "completion_conditions": ["当前页面唯一的多行输入框可见"],
                    "execution_class": "observe",
                    "goal_entities": {
                        "input_text": "first line\nsecond line",
                        "target_ui_label": "多行输入框",
                    },
                }
            },
        }

        scene = GenericSceneObserver(provider).observe(
            frames=stable_frames(),
            goal_context=context,
        )

        self.assertEqual(3, provider.calls)
        self.assertEqual(
            "local_audited_input_1",
            scene.unique_trusted_goal_element().element_id,
        )

    def test_non_input_subgoal_cannot_reuse_future_input_crop_hint(self) -> None:
        context = {
            "objective": "输入文本后刷新页面",
            "entities": {
                "input_text": "future text",
                "active_subgoal_visual_context": {
                    "subgoal_id": "reload",
                    "objective": "刷新当前页面",
                    "constraints": [],
                    "completion_conditions": ["页面刷新完成"],
                    "execution_class": "navigate",
                    "goal_entities": {"input_text": "future text"},
                },
            },
        }

        self.assertIsNone(
            _input_audit_retry_roi(
                context,
                preliminary_input_bounds_hint=(130, 590, 870, 740),
                first_audit_raw=json.dumps(
                    input_audit_payload(application_inputs=[]),
                    ensure_ascii=False,
                ),
            )
        )

    def test_empty_input_audit_retry_keeps_keyboard_panned_field_in_crop(
        self,
    ) -> None:
        first = scene_payload()
        first["elements"][0].update(
            {
                "role": "input",
                "meaning": "application_text_input",
                "label": "长文本",
                "bounds": [135, 590, 860, 690],
                "states": {
                    "goal_relevant": True,
                    "fully_visible": True,
                    "focused": True,
                    "value": "",
                    "soft_keyboard_visible": True,
                },
            }
        )
        first_keyboard = {
            "visible": True,
            "bounds": [40, 580, 960, 930],
            "layout": "qwerty",
            "input_mode": "direct_latin",
            "mode_switch": None,
        }
        recovered_keyboard = {
            "visible": True,
            "bounds": [40, 430, 960, 950],
            "layout": "qwerty",
            "input_mode": "direct_latin",
            "mode_switch": None,
        }
        recovered = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="long-text-field",
                    bounds=[130, 80, 870, 230],
                    text="",
                    placeholder="长文本",
                    field_labels=["长文本"],
                )
            ],
            keyboard=recovered_keyboard,
        )
        provider = SequenceProvider(
            [
                first,
                input_audit_payload(
                    application_inputs=[],
                    keyboard=first_keyboard,
                ),
                recovered,
            ]
        )
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={
                "objective": "在唯一长文本输入框输入 alphabet continuation",
                "entities": {
                    "input_text": "alphabet continuation",
                    "active_subgoal_visual_context": {
                        "subgoal_id": "input_text",
                        "objective": "在唯一长文本输入框输入 alphabet continuation",
                        "constraints": [],
                        "completion_conditions": [
                            "长文本输入框逐字等于 alphabet continuation"
                        ],
                        "execution_class": "navigate",
                        "goal_entities": {
                            "input_text": "alphabet continuation",
                            "active_input_transaction_text": "alphabet continuation",
                            "active_input_field_id": "input_field_1",
                            "active_input_field_label": "长文本",
                            "active_input_multiline": False,
                        },
                    },
                },
            },
        )

        self.assertEqual(3, provider.calls)
        target = scene.unique_trusted_goal_element()
        self.assertIsNotNone(target)
        self.assertEqual("input_field_1", target.states["input_field_id"])
        self.assertEqual((0.13, 0.356, 0.87, 0.461), target.bounds)
        retry_prompt = provider.messages_seen[2][1]["content"][0]["text"]
        self.assertIn("[0, 300, 1000, 1000]", retry_prompt)
        self.assertTrue(observer.last_diagnostics["input_structure_audit_retry_used"])

    def test_input_audit_retry_keyboard_hint_varies_and_remains_fail_closed(
        self,
    ) -> None:
        context = {
            "objective": "在备注输入框输入 release candidate",
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "type_notes",
                    "objective": "在备注输入框输入 release candidate",
                    "constraints": [],
                    "completion_conditions": [
                        "备注输入框逐字等于 release candidate"
                    ],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "release candidate",
                        "active_input_field_id": "notes_field",
                        "active_input_field_label": "备注",
                    }
                }
            },
        }
        keyboard_audit = input_audit_payload(
            application_inputs=[],
            keyboard={
                "visible": True,
                "bounds": [50, 650, 950, 980],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "mode_switch": None,
            },
        )

        self.assertEqual(
            (0, 386, 1000, 1000),
            _input_audit_retry_roi(
                context,
                preliminary_input_bounds_hint=(120, 720, 880, 810),
                first_audit_raw=json.dumps(keyboard_audit, ensure_ascii=False),
            ),
        )
        invalid_keyboard = {
            **keyboard_audit,
            "keyboard": {
                **keyboard_audit["keyboard"],
                "bounds": [50, 650, 200, 760],
            },
        }
        self.assertEqual(
            (0, 480, 1000, 1000),
            _input_audit_retry_roi(
                context,
                preliminary_input_bounds_hint=(120, 720, 880, 810),
                first_audit_raw=json.dumps(invalid_keyboard, ensure_ascii=False),
            ),
        )
        self.assertEqual(
            (0, 480, 1000, 1000),
            _input_audit_retry_roi(
                context,
                preliminary_input_bounds_hint=(120, 720, 880, 810),
                first_audit_raw="not-json",
            ),
        )
        self.assertEqual(
            (0, 130, 1000, 1000),
            _input_audit_retry_roi(
                context,
                preliminary_input_bounds_hint=(135, 490, 865, 730),
                first_audit_raw="not-json",
            ),
        )

    def test_active_input_transaction_fails_as_observation_when_two_audits_are_empty(
        self,
    ) -> None:
        first = scene_payload()
        first["elements"][0]["role"] = "input"
        first["elements"][0]["meaning"] = "message_input"
        first["elements"][0]["bounds"] = [120, 430, 880, 520]
        first["elements"][0]["states"] = {
            "goal_relevant": True,
            "fully_visible": True,
            "focused": True,
            "value": "",
        }
        context = {
            "objective": "在唯一输入框输入 alpha",
            "entities": {
                "input_text": "alpha",
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_text",
                    "objective": "在唯一输入框输入 alpha",
                    "constraints": [],
                    "completion_conditions": ["输入框逐字等于 alpha"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "input_text": "alpha",
                        "active_input_transaction_text": "alpha",
                    },
                },
            },
        }

        with self.assertRaisesRegex(
            VisionAgentError,
            "专用输入结构审计没有建立当前输入事务的唯一本地目标",
        ):
            GenericSceneObserver(
                SequenceProvider(
                    [
                        first,
                        input_audit_payload(application_inputs=[]),
                        input_audit_payload(application_inputs=[]),
                    ]
                )
            ).observe(frames=stable_frames(), goal_context=context)

    def test_confirmation_can_return_trusted_input_when_local_auxiliary_is_omitted(
        self,
    ) -> None:
        first = scene_payload()
        first["elements"][0].update(
            {
                "role": "input",
                "meaning": "application_text_input",
                "label": "first",
                "bounds": [140, 285, 860, 450],
                "states": {
                    "goal_relevant": True,
                    "fully_visible": True,
                    "focused": True,
                    "value": "first",
                },
                "evidence": ["正文", "first", "caret"],
            }
        )
        context = {
            "objective": "使当前唯一多行输入框内容精确等于授权文字",
            "entities": {
                "input_text": "first\nsecond",
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_exact_text",
                    "objective": "在当前唯一输入框中逐字输入授权文字",
                    "constraints": ["不要发送或提交"],
                    "completion_conditions": ["输入框逐字等于授权文字"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "input_text": "first\nsecond",
                        "active_input_transaction_text": "first\nsecond",
                        "active_input_field_id": "input_field_1",
                        "active_input_field_label": "正文",
                        "active_input_multiline": True,
                    },
                },
            },
            "_allow_omitted_local_input_auxiliary_confirmation": True,
        }
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    bounds=[140, 285, 860, 450],
                    text="first",
                    field_labels=["正文"],
                    visible_editable_cues=["完整边框", "焦点高亮"],
                )
            ],
            keyboard={
                "visible": True,
                "bounds": [100, 600, 900, 1000],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "case_mode": "lower",
                "mode_switch": None,
                "enter_key": {
                    "label": "",
                    "bounds": [820, 920, 900, 985],
                    "confidence": 1.0,
                    "fully_visible": True,
                    "key_action": "newline",
                },
            },
        )

        observed = GenericSceneObserver(
            SequenceProvider([first, audit, audit])
        ).observe(frames=stable_frames(), goal_context=context)

        trusted_input = observed.get_element("local_audited_input_1")
        self.assertEqual("first", trusted_input.states["value"])
        self.assertEqual(
            "input_field_1",
            trusted_input.states["input_field_id"],
        )
        self.assertFalse(
            any(
                element.meaning == "input_exact_enter_key"
                for element in observed.elements
            )
        )

    def test_system_home_direction_audit_uses_privacy_minimized_views(self):
        provider = FakeProvider(
            {
                "protocol_version": ORIENTATION_AUDIT_PROTOCOL_VERSION,
                "phone_content_rotation": "upright",
                "confidence": 0.95,
                "evidence": ["底部手机系统导航结构位于画布下缘"],
            }
        )
        observer = GenericSceneObserver(provider)
        frames = stable_frames()

        credential = observer.audit_coordinate_free_system_navigation_alignment(
            frames=frames,
            device_id="device-a",
            scene_fingerprint="scene-a",
        )

        self.assertEqual("upright", credential.phone_content_rotation)
        self.assertEqual(1, provider.calls)
        self.assertEqual(
            "2026-08-19-system-navigation-privacy-view-v1",
            observer.last_orientation_audit_diagnostics["privacy_view_version"],
        )

    def test_two_empty_input_audits_remain_fail_closed(self) -> None:
        first = scene_payload()
        first["elements"][0]["role"] = "input"
        first["elements"][0]["meaning"] = "message_input"
        first["elements"][0]["states"] = {"goal_relevant": True}
        first["elements"][0]["bounds"] = [60, 1750, 940, 1880]
        empty_audit = input_audit_payload(application_inputs=[])
        provider = SequenceProvider([first, empty_audit, empty_audit])
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={
                "objective": "在底部唯一输入框中保留 codex 草稿",
                "entities": {"input_text": "codex", "spatial_hint": "bottom"},
            },
        )

        self.assertEqual(3, provider.calls)
        self.assertIsNone(scene.unique_trusted_goal_element())
        self.assertTrue(
            observer.last_diagnostics["input_structure_audit_retry_used"]
        )

    def test_crop_local_input_audit_rejects_internal_edge_contact(self) -> None:
        payload = input_audit_payload(
            application_inputs=[
                audited_application_input(bounds=[0, 700, 600, 900])
            ]
        )

        with self.assertRaisesRegex(VisionAgentError, "crop 内部边界"):
            _map_input_structure_crop_audit_to_full(
                json.dumps(payload, ensure_ascii=False),
                roi_bounds=(100, 580, 900, 1000),
            )

    def test_input_geometry_with_action_field_cannot_be_isolated(self) -> None:
        first = scene_payload()
        first["elements"][0].update(
            {
                "role": "input",
                "meaning": "message_input",
                "bounds": [60, 1750, 940, 1880],
                "states": {"goal_relevant": True},
                "tap": True,
            }
        )
        observer = GenericSceneObserver(SequenceProvider([first]))

        with self.assertRaisesRegex(VisionAgentError, "bounds|protocol|动作字段"):
            observer.observe(
                frames=stable_frames(),
                goal_context={
                    "objective": "输入框内容为 codex 且不提交",
                    "entities": {"input_text": "codex"},
                },
            )

    def test_input_isolation_preserves_page_identity_but_not_peripheral_geometry(self) -> None:
        compact = scene_payload()
        compact["foreground_app_id"] = "chat_app"
        compact["screen_id"] = "draft_chat"
        compact["elements"] = [
            {
                "element_id": "page-title",
                "role": "text",
                "meaning": "page_title",
                "label": "本机草稿页",
                "bounds": [340, 20, 660, 70],
                "confidence": 1.0,
                "states": {"goal_relevant": False, "fully_visible": True},
                "evidence": ["顶部主标题清晰可见"],
            },
            {
                "element_id": "model-input",
                "role": "input",
                "meaning": "message_input_box",
                "label": "",
                "bounds": [120, 1380, 680, 1460],
                "confidence": 1.0,
                "states": {"goal_relevant": True, "fully_visible": True, "value": ""},
                "evidence": ["底部输入区域"],
            },
            {
                "element_id": "peripheral-icon",
                "role": "icon",
                "meaning": "more_options",
                "label": "更多",
                "bounds": [880, 1390, 960, 1450],
                "confidence": 1.0,
                "states": {"goal_relevant": False, "fully_visible": True},
                "evidence": ["输入区域旁的图标"],
            },
        ]
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(text="", placeholder="写点什么")
            ]
        )
        provider = SequenceProvider([compact, audit])

        scene = GenericSceneObserver(provider).observe(
            frames=stable_frames(),
            goal_context={
                "objective": "本机草稿页的底部唯一输入框内容为 codex",
                "entities": {
                    "target_ui_label": "本机草稿页",
                    "input_text": "codex",
                },
            },
        )

        self.assertEqual(2, provider.calls)
        self.assertEqual("draft_chat", scene.screen_id)
        self.assertEqual(
            ["page-title", "local_audited_input_1"],
            [item.element_id for item in scene.elements],
        )
        self.assertEqual(
            "local_audited_input_1",
            scene.unique_trusted_goal_element().element_id,
        )

    def test_targeted_delta_preserves_compact_authority_and_merges_only_evidence(self) -> None:
        base = _parse_scene(
            json.dumps(scene_payload(), ensure_ascii=False),
            fingerprint="local-fingerprint",
        )
        element = dict(scene_payload()["elements"][0])
        element.update(
            {
                "element_id": "target-1",
                "meaning": "open_target",
                "label": "目标",
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": ["目标按钮四边完整可见"],
            }
        )
        scene = _parse_targeted_scene_delta(
            json.dumps(
                targeted_delta_payload(
                    elements=[element],
                    summary_addendum="底部存在页面延续标记",
                    confidence=0.91,
                ),
                ensure_ascii=False,
            ),
            base_scene=base,
            fingerprint="local-fingerprint",
            goal_context={"objective": "查看目标"},
        )

        self.assertEqual(base.app_id, scene.app_id)
        self.assertEqual(base.screen_id, scene.screen_id)
        self.assertEqual(base.system_ui, scene.system_ui)
        self.assertEqual(base.camera_alignment, scene.camera_alignment)
        self.assertEqual(base.overlays, scene.overlays)
        self.assertEqual(base.stable, scene.stable)
        self.assertEqual("local-fingerprint", scene.fingerprint)
        self.assertEqual("target-1", scene.elements[0].element_id)
        self.assertEqual(0.91, scene.confidence)
        self.assertEqual("计算器首页；底部存在页面延续标记", scene.summary)

    def test_targeted_delta_normalizes_only_single_evidence_string_shorthand(self) -> None:
        base = _parse_scene(
            json.dumps(scene_payload(), ensure_ascii=False),
            fingerprint="local-fingerprint",
        )
        first = dict(scene_payload()["elements"][0])
        first.update(
            {
                "element_id": "field",
                "role": "input",
                "meaning": "editable_text_field",
                "bounds": [100, 100, 800, 220],
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": "唯一输入区域四边完整可见",
            }
        )
        second = dict(scene_payload()["elements"][0])
        second.update(
            {
                "element_id": "cancel",
                "meaning": "cancel",
                "label": "取消",
                "bounds": [820, 100, 940, 220],
                "states": {"goal_relevant": False, "fully_visible": True},
                "evidence": "右侧取消按钮",
            }
        )
        scene = _parse_targeted_scene_delta(
            json.dumps(
                targeted_delta_payload(elements=[first, second]),
                ensure_ascii=False,
            ),
            base_scene=base,
            fingerprint="local-fingerprint",
        )

        self.assertEqual(
            ("唯一输入区域四边完整可见",),
            scene.elements[0].evidence,
        )
        self.assertEqual(("右侧取消按钮",), scene.elements[1].evidence)

        for invalid_evidence in ({"text": "事实"}, 7):
            invalid = targeted_delta_payload(
                elements=[{**first, "evidence": invalid_evidence}]
            )
            with self.subTest(evidence=invalid_evidence), self.assertRaisesRegex(
                VisionAgentError, "最小增量协议"
            ):
                _parse_targeted_scene_delta(
                    json.dumps(invalid, ensure_ascii=False),
                    base_scene=base,
                    fingerprint="local-fingerprint",
                )

    def test_targeted_delta_normalizes_only_exact_valid_bounds_shorthand(self) -> None:
        base = _parse_scene(
            json.dumps(scene_payload(), ensure_ascii=False),
            fingerprint="local-fingerprint",
        )
        element = dict(scene_payload()["elements"][0])
        element.update(
            {
                "element_id": "draft_input_box",
                "role": "input",
                "meaning": "unique_temporary_draft_area",
                "label": "lxs,",
                "bounds": {"x": 145, "y": 535, "w": 560, "h": 45},
                "confidence": 1.0,
                "states": {
                    "fully_visible": True,
                    "goal_relevant": True,
                    "value": "lxs,",
                    "keyboard_layout": "qwerty",
                    "keyboard_input_mode": "direct_latin",
                },
                "evidence": "输入框逐字显示 lxs,",
            }
        )

        valid_bounds = (
            {"x": 145, "y": 535, "w": 560, "h": 45},
            {"x": 145, "y": 535, "width": 560, "height": 45},
            {"x1": 145, "y1": 535, "x2": 705, "y2": 580},
        )
        for bounds in valid_bounds:
            with self.subTest(valid_bounds=bounds):
                parsed = _parse_targeted_scene_delta(
                    json.dumps(
                        targeted_delta_payload(
                            elements=[{**element, "bounds": bounds}],
                            confidence=1.0,
                        ),
                        ensure_ascii=False,
                    ),
                    base_scene=base,
                    fingerprint="local-fingerprint",
                )

                self.assertEqual(
                    (0.145, 0.535, 0.705, 0.58),
                    parsed.elements[0].bounds,
                )
                self.assertEqual("lxs,", parsed.elements[0].states["value"])

        for invalid_bounds in (
            {"x": 145, "y": 535, "w": 560},
            {"x": 145, "y": 535, "w": 560, "h": 45, "right": 705},
            {"x": 145, "y": 535, "w": -1, "h": 45},
            {"x": 900, "y": 535, "w": 560, "h": 45},
            {"x": True, "y": 535, "w": 560, "h": 45},
            {"x": 145, "y": 535, "w": 560, "height": 45},
            {"x": 145, "y": 535, "width": 560, "height": 45, "r": 705},
            {"x": 145, "y": 535, "width": "560", "height": 45},
            {"x": 145, "y": 535, "width": 560, "height": 0},
            {"x1": 145, "y1": 535, "x2": 145, "y2": 580},
            {"x1": 145, "y1": 535, "x2": 705, "y2": 1001},
            {"x1": 145, "y1": 535, "x2": "705", "y2": 580},
            {"x1": 145, "y1": 535, "x2": 705},
        ):
            with self.subTest(bounds=invalid_bounds), self.assertRaisesRegex(
                VisionAgentError,
                "最小增量协议",
            ):
                invalid = targeted_delta_payload(
                    elements=[{**element, "bounds": invalid_bounds}]
                )
                _parse_targeted_scene_delta(
                    json.dumps(invalid, ensure_ascii=False),
                    base_scene=base,
                    fingerprint="local-fingerprint",
                )

    def test_targeted_delta_reattaches_unknown_xywh_canvas_only_to_unique_base_identity(self) -> None:
        base_payload = scene_payload()
        base_payload["elements"] = [
            {
                "element_id": "bottom_nav_window_btn",
                "role": "button",
                "meaning": "window_button",
                "label": "窗口",
                "bounds": [390, 880, 510, 980],
                "confidence": 1.0,
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": ["底部窗口按钮完整可见"],
            }
        ]
        base = _parse_scene(
            json.dumps(base_payload, ensure_ascii=False),
            fingerprint="local-fingerprint",
        )
        live_element = {
            "element_id": "window_button_01",
            "role": "button",
            "meaning": "window_button",
            "label": "窗口",
            "bounds": {"x": 390, "y": 1750, "w": 120, "h": 140},
            "confidence": 1.0,
            "states": {"goal_relevant": True, "fully_visible": True},
            "evidence": "底部导航栏可见窗口按钮",
        }

        parsed = _parse_targeted_scene_delta(
            json.dumps(
                targeted_delta_payload(elements=[live_element], confidence=1.0),
                ensure_ascii=False,
            ),
            base_scene=base,
            fingerprint="local-fingerprint",
        )

        self.assertEqual("bottom_nav_window_btn", parsed.elements[0].element_id)
        self.assertEqual((0.39, 0.88, 0.51, 0.98), parsed.elements[0].bounds)

        rejection_variants = []
        no_match_payload = scene_payload()
        no_match_payload["elements"] = []
        rejection_variants.append(
            _parse_scene(
                json.dumps(no_match_payload, ensure_ascii=False),
                fingerprint="local-fingerprint",
            )
        )
        duplicate_payload = dict(base_payload)
        duplicate_payload["elements"] = [
            base_payload["elements"][0],
            {**base_payload["elements"][0], "element_id": "other_window"},
        ]
        rejection_variants.append(
            _parse_scene(
                json.dumps(duplicate_payload, ensure_ascii=False),
                fingerprint="local-fingerprint",
            )
        )
        clipped_payload = dict(base_payload)
        clipped_payload["elements"] = [
            {
                **base_payload["elements"][0],
                "states": {"goal_relevant": True, "fully_visible": False},
            }
        ]
        rejection_variants.append(
            _parse_scene(
                json.dumps(clipped_payload, ensure_ascii=False),
                fingerprint="local-fingerprint",
            )
        )
        low_confidence_payload = dict(base_payload)
        low_confidence_payload["elements"] = [
            {**base_payload["elements"][0], "confidence": 0.89}
        ]
        rejection_variants.append(
            _parse_scene(
                json.dumps(low_confidence_payload, ensure_ascii=False),
                fingerprint="local-fingerprint",
            )
        )
        for unsafe_base in rejection_variants:
            with self.subTest(base=unsafe_base), self.assertRaisesRegex(
                VisionAgentError,
                "最小增量协议",
            ):
                _parse_targeted_scene_delta(
                    json.dumps(
                        targeted_delta_payload(elements=[live_element]),
                        ensure_ascii=False,
                    ),
                    base_scene=unsafe_base,
                    fingerprint="local-fingerprint",
                )

        unrelated_live_element = {
            **live_element,
            "element_id": "container_feed_001",
            "role": "container",
            "meaning": "content_list",
            "label": "新闻信息流列表",
            "bounds": {"x": 50, "y": 130, "w": 900, "h": 1400},
            "states": {
                "goal_relevant": True,
                "fully_visible": False,
                "scrollable": True,
                "scroll_axis": "vertical",
            },
        }
        with self.assertRaisesRegex(VisionAgentError, "最小增量协议"):
            _parse_targeted_scene_delta(
                json.dumps(
                    targeted_delta_payload(elements=[unrelated_live_element]),
                    ensure_ascii=False,
                ),
                base_scene=base,
                fingerprint="local-fingerprint",
            )

    def test_targeted_delta_rejects_live_unknown_portrait_canvas_shapes(self) -> None:
        base_payload = scene_payload()
        base_payload["elements"] = []
        base = _parse_scene(
            json.dumps(base_payload, ensure_ascii=False),
            fingerprint="local-fingerprint",
        )
        live_shapes = (
            {
                "element_id": "sys_nav_back_btn",
                "role": "button",
                "meaning": "back",
                "label": "<",
                "bounds": {"x": 130, "y": 1950, "w": 80, "h": 80},
                "confidence": 0.95,
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": "底部系统导航栏左侧可见返回图标",
            },
            {
                "element_id": "container_list_view",
                "role": "container",
                "meaning": "list_container",
                "label": "模式选择列表",
                "bounds": {"x": 100, "y": 50, "w": 800, "h": 1400},
                "confidence": 1.0,
                "states": {
                    "goal_relevant": True,
                    "fully_visible": False,
                    "scrollable": True,
                    "scroll_axis": "vertical",
                },
                "evidence": "底部边缘可见内容被截断",
            },
        )
        for element in live_shapes:
            with self.subTest(element_id=element["element_id"]), self.assertRaisesRegex(
                VisionAgentError,
                "最小增量协议",
            ):
                _parse_targeted_scene_delta(
                    json.dumps(
                        targeted_delta_payload(elements=[element]),
                        ensure_ascii=False,
                    ),
                    base_scene=base,
                    fingerprint="local-fingerprint",
                )

    def test_targeted_delta_discards_only_explicit_non_goal_out_of_range_peripheral(self) -> None:
        base = _parse_scene(
            json.dumps(scene_payload(), ensure_ascii=False),
            fingerprint="local-fingerprint",
        )
        target = dict(scene_payload()["elements"][0])
        target.update(
            {
                "element_id": "search_input_clipped",
                "role": "input",
                "meaning": "search_field",
                "label": "",
                "bounds": {"x": 0, "y": 0, "w": 1000, "h": 45},
                "states": {
                    "goal_relevant": True,
                    "fully_visible": False,
                    "value": "",
                },
                "evidence": "顶部仅见输入框边缘",
            }
        )
        peripheral = dict(scene_payload()["elements"][0])
        peripheral.update(
            {
                "element_id": "keyboard_switch_hint",
                "role": "button",
                "meaning": "switch_keyboard_input_mode",
                "label": "英",
                "bounds": {"x": 735, "y": 2265, "w": 90, "h": 110},
                "states": {
                    "goal_relevant": False,
                    "fully_visible": True,
                },
                "evidence": "模型误用了源像素纵坐标",
            }
        )

        parsed = _parse_targeted_scene_delta(
            json.dumps(
                targeted_delta_payload(elements=[target, peripheral]),
                ensure_ascii=False,
            ),
            base_scene=base,
            fingerprint="local-fingerprint",
            goal_context={"objective": "确认搜索输入框状态"},
        )
        self.assertEqual(
            ["search_input_clipped"],
            [element.element_id for element in parsed.elements],
        )

        unsafe = targeted_delta_payload(
            elements=[
                target,
                {
                    **peripheral,
                    "states": {
                        "goal_relevant": True,
                        "fully_visible": True,
                    },
                },
            ]
        )
        with self.assertRaisesRegex(VisionAgentError, "最小增量协议"):
            _parse_targeted_scene_delta(
                json.dumps(unsafe, ensure_ascii=False),
                base_scene=base,
                fingerprint="local-fingerprint",
                goal_context={"objective": "确认搜索输入框状态"},
            )

    def test_targeted_delta_rejects_legacy_scene_and_authority_fields(self) -> None:
        base = _parse_scene(
            json.dumps(scene_payload(), ensure_ascii=False),
            fingerprint="local-fingerprint",
        )
        invalid_payloads = [scene_payload()]
        for field_name in (
            "foreground_app_id",
            "screen_id",
            "system_ui",
            "camera_alignment",
            "overlays",
            "stable",
            "fingerprint",
        ):
            payload = targeted_delta_payload()
            payload[field_name] = "forbidden"
            invalid_payloads.append(payload)

        for payload in invalid_payloads:
            with self.subTest(fields=sorted(payload)), self.assertRaisesRegex(
                VisionAgentError, "最小增量协议"
            ):
                _parse_targeted_scene_delta(
                    json.dumps(payload, ensure_ascii=False),
                    base_scene=base,
                    fingerprint="local-fingerprint",
                )

    def test_targeted_delta_rejects_duplicates_and_more_than_twelve_elements(self) -> None:
        base = _parse_scene(
            json.dumps(scene_payload(), ensure_ascii=False),
            fingerprint="local-fingerprint",
        )
        duplicate = (
            '{"protocol_version":"'
            + TARGETED_SCENE_DELTA_PROTOCOL_VERSION
            + '","summary_addendum":"","elements":[],"confidence":0.9,'
            '"confidence":0.8}'
        )
        with self.assertRaisesRegex(VisionAgentError, "重复 JSON 字段"):
            _parse_targeted_scene_delta(
                duplicate,
                base_scene=base,
                fingerprint="local-fingerprint",
            )

        element = scene_payload()["elements"][0]
        oversized = targeted_delta_payload(
            elements=[
                {**element, "element_id": f"e{index}"}
                for index in range(13)
            ]
        )
        with self.assertRaisesRegex(VisionAgentError, "最小增量协议"):
            _parse_targeted_scene_delta(
                json.dumps(oversized, ensure_ascii=False),
                base_scene=base,
                fingerprint="local-fingerprint",
            )

    def test_empty_targeted_delta_is_safe_and_prompt_forbids_scene_repetition(self) -> None:
        base = _parse_scene(
            json.dumps(scene_payload(), ensure_ascii=False),
            fingerprint="local-fingerprint",
        )
        scene = _parse_targeted_scene_delta(
            json.dumps(targeted_delta_payload(), ensure_ascii=False),
            base_scene=base,
            fingerprint="local-fingerprint",
        )
        prompt = _targeted_prompt({}, first_scene=base.to_dict())

        self.assertEqual((), scene.elements)
        self.assertEqual(base.summary, scene.summary)
        self.assertIn(TARGETED_SCENE_DELTA_PROTOCOL_VERSION, prompt)
        self.assertIn("四个字段缺一不可", prompt)
        self.assertIn("不要重复或返回foreground_app_id", prompt)
        self.assertNotIn(f'"protocol_version":"{UI_SCENE_PROTOCOL_VERSION}"', prompt)

    def test_goal_element_without_visible_evidence_triggers_targeted_refinement(self) -> None:
        first = scene_payload()
        first["screen_id"] = "generic_acceptance"
        first["elements"][0].update(
            {
                "role": "button",
                "meaning": "back_to_list",
                "label": "返回验收模式选择",
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": [],
            }
        )
        refined_element = json.loads(json.dumps(first["elements"][0], ensure_ascii=False))
        refined_element["element_id"] = "return_entry_01"
        refined_element["evidence"] = [
            "说明文字下方带下划线的白色返回入口，四边完整可见"
        ]
        refined = targeted_delta_payload(elements=[refined_element])
        provider = SequenceProvider([first, refined])
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={
                "objective": "回到验收模式选择列表",
                "entities": {"target_ui_label": "返回验收模式选择"},
            },
        )

        self.assertEqual(2, provider.calls)
        self.assertTrue(observer.last_diagnostics["targeted_refinement_used"])
        self.assertEqual(
            ("说明文字下方带下划线的白色返回入口，四边完整可见",),
            scene.elements[0].evidence,
        )

    def test_unrelated_goal_relevant_element_cannot_suppress_exact_label_refinement(self) -> None:
        first = scene_payload()
        first["screen_id"] = "acceptance_modes"
        first["summary"] = "验收模式列表"
        first["elements"][0].update(
            {
                "meaning": "status_display",
                "label": "等待动作",
                "states": {"goal_relevant": True, "fully_visible": True},
            }
        )
        refined = targeted_delta_payload(
            summary_addendum="底部边缘存在部分可见的后续内容，列表仍在延伸"
        )
        provider = SequenceProvider([first, refined])
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={
                "objective": "查看连续闭环结果",
                "entities": {"target_ui_label": "连续闭环"},
            },
        )

        self.assertEqual(2, provider.calls)
        self.assertTrue(observer.last_diagnostics["targeted_refinement_used"])
        self.assertEqual([], list(scene.elements))
        self.assertIn("部分可见的后续内容", scene.summary)

    def test_all_observation_prompts_recognize_prefilled_inputs_without_authorizing_submit(self) -> None:
        empty = scene_payload()
        empty["elements"] = []
        provider = SequenceProvider([empty, empty, input_audit_payload()])
        observer = GenericSceneObserver(provider)

        observer.observe(
            frames=stable_frames(),
            goal_context={"objective": "把当前输入框的文字改为Agent123"},
        )

        compact_text = provider.messages_seen[0][1]["content"][0]["text"]
        targeted_text = provider.messages_seen[1][1]["content"][0]["text"]
        for prompt in (compact_text, targeted_text):
            self.assertIn("输入框可能为空，也可能已经含有文字", prompt)
            self.assertIn("预填充且未聚焦时可以没有光标", prompt)
            self.assertIn("role=input", prompt)
            self.assertIn("不得仅因没有光标而降级成text或container", prompt)
            self.assertIn("尾部功能控件必须作为另一个控件观察", prompt)
            self.assertIn("绝不表示可以激活尾部控件", prompt)
            self.assertIn("框内文字的内容或主题不能改变控件角色", prompt)

    def test_editable_field_wording_triggers_generic_input_structure_audit(self) -> None:
        empty = scene_payload()
        empty["elements"] = []
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="field-with-scan",
                    right_button={
                        "label": "扫描",
                        "bounds": [780, 40, 850, 110],
                        "confidence": 0.97,
                    },
                )
            ]
        )
        provider = SequenceProvider([empty, empty, audit])
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={"objective": "使顶部白色字段进入编辑焦点并显示软键盘"},
        )

        candidate = scene.unique_trusted_goal_element()
        self.assertIsNotNone(candidate)
        self.assertEqual("input", candidate.role)
        self.assertEqual("已有文字", candidate.label)
        self.assertEqual((0.11, 0.04, 0.78, 0.11), candidate.bounds)
        self.assertTrue(observer.last_diagnostics["input_structure_audit_used"])
        audit_text = provider.messages_seen[2][1]["content"][0]["text"]
        self.assertIn("trailing utility control", audit_text)
        self.assertIn("is never authorized for activation", audit_text)

    def test_input_audit_recovers_empty_top_application_input_and_keyboard_facts(self) -> None:
        empty = scene_payload()
        empty["summary"] = "顶部区域和软键盘清楚，但快速观察没有建立控件"
        empty["elements"] = []
        keyboard = {
            "visible": True,
            "bounds": [0, 360, 1000, 1000],
            "layout": "qwerty",
            "input_mode": "chinese_pinyin",
            "mode_switch": {
                "label": "中",
                "bounds": [650, 900, 760, 970],
                "confidence": 0.97,
                "current_mode": "chinese_pinyin",
                "target_mode": "direct_latin",
            },
        }
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="empty-top-input",
                    bounds=[80, 35, 820, 115],
                    text="",
                    placeholder="搜索",
                    right_button=None,
                )
            ],
            keyboard=keyboard,
        )
        provider = SequenceProvider([empty, empty, audit])
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={"objective": "读取顶部空输入框及当前键盘输入模式"},
        )

        candidate = scene.unique_trusted_goal_element()
        self.assertIsNotNone(candidate)
        self.assertEqual("input", candidate.role)
        self.assertEqual("", candidate.states["value"])
        self.assertEqual("搜索", candidate.states["placeholder"])
        self.assertTrue(candidate.states["focused"])
        self.assertEqual("qwerty", candidate.states["keyboard_layout"])
        self.assertEqual("chinese_pinyin", candidate.states["keyboard_input_mode"])
        mode_switch = scene.get_element("local_audited_keyboard_mode_switch_1")
        self.assertFalse(mode_switch.states["goal_relevant"])
        self.assertEqual("direct_latin", mode_switch.states["target_mode"])

    def test_input_authorization_requires_current_qwerty_anchors(self) -> None:
        empty = scene_payload()
        empty["elements"] = []
        audit = input_audit_payload(
            application_inputs=[audited_application_input(text="", placeholder="输入")],
            keyboard={
                "visible": True,
                "bounds": [0, 360, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "qwerty_anchors": None,
                "mode_switch": None,
            },
        )

        with self.assertRaisesRegex(VisionAgentError, "QWERTY anchors"):
            GenericSceneObserver(SequenceProvider([empty, empty, audit])).observe(
                frames=stable_frames(),
                goal_context={
                    "objective": "在唯一输入框输入 agent",
                    "entities": {"input_text": "agent"},
                },
            )

    def test_input_authorization_binds_locally_validated_qwerty_geometry(self) -> None:
        empty = scene_payload()
        empty["elements"] = []
        audit = input_audit_payload(
            application_inputs=[audited_application_input(text="", placeholder="输入")],
            keyboard={
                "visible": True,
                "bounds": [0, 360, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "mode_switch": None,
            },
        )

        scene = GenericSceneObserver(SequenceProvider([empty, empty, audit])).observe(
            frames=stable_frames(),
            goal_context={"objective": "在唯一输入框输入 agent"},
        )

        geometry = scene.unique_trusted_goal_element().states["keyboard_geometry"]
        self.assertEqual("qwerty", geometry["type"])
        self.assertEqual("input_structure_audit", geometry["source"])
        self.assertEqual({"q", "p", "a", "l", "z", "m", "backspace"}, set(geometry["anchors"]))

    def test_input_audit_normalizes_symbols_layout_without_relaxing_schema(self) -> None:
        for observed_layout in (" Symbols ", "symbol_grid", "qwerty_symbol"):
            with self.subTest(observed_layout=observed_layout):
                empty = scene_payload()
                empty["elements"] = []
                audit = input_audit_payload(
                    application_inputs=[audited_application_input(text=".com")],
                    keyboard={
                        "visible": True,
                        "bounds": [0, 360, 1000, 1000],
                        "layout": observed_layout,
                        "input_mode": "chinese_pinyin",
                        "mode_switch": None,
                        "layout_switches": [
                            {
                                "label": "返回",
                                "bounds": [60, 880, 200, 940],
                                "confidence": 1.0,
                                "current_layout": observed_layout,
                                "target_layout": "qwerty",
                            }
                        ],
                    },
                )
                observer = GenericSceneObserver(
                    SequenceProvider([empty, empty, audit])
                )

                scene = observer.observe(
                    frames=stable_frames(),
                    goal_context={"objective": "确认输入框内容已经是 .com"},
                )

                candidate = scene.unique_trusted_goal_element()
                self.assertIsNotNone(candidate)
                self.assertEqual("symbol", candidate.states["keyboard_layout"])

    def test_input_audit_infers_composite_symbol_only_from_unique_next_key(self) -> None:
        current_value = "longinputvalidation2026"
        target_value = current_value + ":"
        base_scene = _parse_scene(
            json.dumps(scene_payload(), ensure_ascii=False),
            fingerprint="frame-composite-symbol",
        )

        def payload(layout: str, literal_keys: list[dict]) -> dict:
            return input_audit_payload(
                application_inputs=[
                    audited_application_input(
                        structure_id="message-field",
                        bounds=[130, 540, 700, 600],
                        text=current_value,
                    )
                ],
                keyboard={
                    "visible": True,
                    "bounds": [0, 660, 1000, 1000],
                    "layout": layout,
                    "input_mode": "direct_latin",
                    "case_mode": "unknown",
                    "qwerty_anchors": None,
                    "mode_switch": None,
                    "backspace_key": None,
                    "case_switch": None,
                    "literal_keys": literal_keys,
                    "layout_switches": [],
                },
            )

        exact_key = {
            "value": ":",
            "label": ":",
            "key_kind": "character",
            "bounds": [440, 750, 560, 810],
            "confidence": 1.0,
            "fully_visible": True,
        }
        for layout in ("numeric_symbol", "numeric_symbol_grid"):
            with self.subTest(layout=layout):
                scene = _apply_input_structure_audit(
                    base_scene,
                    json.dumps(payload(layout, [exact_key]), ensure_ascii=False),
                    fingerprint="frame-composite-symbol",
                    goal_context={
                        "objective": "输入框逐字等于目标且不发送",
                        "entities": {"input_text": target_value},
                    },
                    ledger_input_value=current_value,
                )
                key = scene.unique_trusted_goal_element()
                self.assertEqual("local_audited_literal_key_1", key.element_id)
                self.assertEqual(":", key.states["key_value"])
                self.assertEqual(
                    "symbol",
                    scene.get_element("local_audited_input_1").states[
                        "keyboard_layout"
                    ],
                )

        for layout, digit in (
            ("numeric_symbol", "1"),
            ("numeric_symbol_grid", "7"),
        ):
            with self.subTest(layout=layout, digit=digit):
                digit_key = {
                    **exact_key,
                    "value": digit,
                    "label": digit,
                }
                scene = _apply_input_structure_audit(
                    base_scene,
                    json.dumps(payload(layout, [digit_key]), ensure_ascii=False),
                    fingerprint="frame-composite-number",
                    goal_context={
                        "objective": "输入框逐字等于目标且不发送",
                        "entities": {"input_text": current_value + digit},
                    },
                    ledger_input_value=current_value,
                )
                key = scene.unique_trusted_goal_element()
                self.assertEqual("local_audited_literal_key_1", key.element_id)
                self.assertEqual(digit, key.states["key_value"])
                self.assertEqual(
                    "symbol",
                    scene.get_element("local_audited_input_1").states[
                        "keyboard_layout"
                    ],
                )

        negative_cases = {
            "unknown_component": payload("symbols_custom", [exact_key]),
            "not_symbol_composite": payload("qwerty_numeric", [exact_key]),
            "mixed": payload("mixed", [exact_key]),
            "missing_key": payload("numeric_symbol", []),
            "wrong_key": payload(
                "numeric_symbol", [{**exact_key, "value": "+", "label": "+"}]
            ),
            "duplicate_key": payload(
                "numeric_symbol",
                [exact_key, {**exact_key, "bounds": [600, 750, 720, 810]}],
            ),
            "low_confidence": payload(
                "numeric_symbol", [{**exact_key, "confidence": 0.5}]
            ),
            "not_whole": payload(
                "numeric_symbol", [{**exact_key, "fully_visible": False}]
            ),
            "outside_keyboard": payload(
                "numeric_symbol", [{**exact_key, "bounds": [440, 500, 560, 560]}]
            ),
        }
        for name, candidate in negative_cases.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(VisionAgentError, "keyboard.layout"):
                    _apply_input_structure_audit(
                        base_scene,
                        json.dumps(candidate, ensure_ascii=False),
                        fingerprint="frame-composite-symbol-negative",
                        goal_context={
                            "objective": "输入框逐字等于目标且不发送",
                            "entities": {"input_text": target_value},
                        },
                        ledger_input_value=current_value,
                    )

        for qwerty_target, key_kind, label in (
            ("a", "character", "a"),
            (" ", "space", "空格"),
        ):
            with self.subTest(qwerty_target=repr(qwerty_target)):
                with self.assertRaisesRegex(VisionAgentError, "keyboard.layout"):
                    _apply_input_structure_audit(
                        base_scene,
                        json.dumps(
                            payload(
                                "numeric_symbol",
                                [
                                    {
                                        **exact_key,
                                        "value": qwerty_target,
                                        "label": label,
                                        "key_kind": key_kind,
                                    }
                                ],
                            ),
                            ensure_ascii=False,
                        ),
                        fingerprint="frame-composite-symbol-qwerty-target",
                        goal_context={
                            "objective": "输入框逐字等于目标且不发送",
                            "entities": {"input_text": current_value + qwerty_target},
                        },
                        ledger_input_value=current_value,
                    )

    def test_scene_normalizes_exact_symbol_grid_layout_alias(self) -> None:
        payload = scene_payload()
        payload["elements"] = [
            {
                "element_id": "input-top",
                "role": "input",
                "meaning": "application_text_input",
                "label": "",
                "bounds": [150, 440, 850, 530],
                "confidence": 0.98,
                "states": {
                    "goal_relevant": True,
                    "value": "2026",
                    "focused": True,
                    "keyboard_layout": "symbol_grid",
                    "keyboard_input_mode": "chinese_pinyin",
                },
                "evidence": ["输入框和符号键盘可见"],
            }
        ]

        scene = _parse_scene(
            json.dumps(payload, ensure_ascii=False),
            fingerprint="scene-symbol-grid-alias",
            goal_context={"objective": "确认输入框内容已经是2026"},
            camera_layout_orientation="portrait",
        )

        self.assertEqual("symbol", scene.elements[0].states["keyboard_layout"])

    def test_input_audit_keeps_unknown_layout_fail_closed(self) -> None:
        for observed_layout in (
            "symbols_custom",
            "numeric_symbol_grid",
            "qwerty_numeric",
            "mixed",
        ):
            with self.subTest(observed_layout=observed_layout):
                empty = scene_payload()
                empty["elements"] = []
                audit = input_audit_payload(
                    keyboard={
                        "visible": True,
                        "bounds": [0, 360, 1000, 1000],
                        "layout": observed_layout,
                        "input_mode": "unknown",
                        "mode_switch": None,
                    }
                )

                with self.assertRaisesRegex(VisionAgentError, "keyboard.layout"):
                    GenericSceneObserver(
                        SequenceProvider([empty, empty, audit])
                    ).observe(
                        frames=stable_frames(),
                        goal_context={"objective": "读取当前输入框和键盘"},
                    )

    def test_input_audit_enriches_known_focused_input_missing_keyboard_facts(self) -> None:
        preliminary = scene_payload()
        preliminary["summary"] = "唯一空输入框已聚焦，外围键盘容器可见"
        preliminary["elements"] = [
            {
                "element_id": "input-target",
                "role": "input",
                "meaning": "application_text_input",
                "label": "",
                "bounds": [150, 440, 850, 530],
                "confidence": 0.98,
                "states": {
                    "value": "",
                    "goal_relevant": True,
                    "focused": True,
                    "fully_visible": True,
                },
                "evidence": ["输入框为空，光标可见"],
            },
            {
                "element_id": "keyboard-container",
                "role": "container",
                "meaning": "keyboard_region",
                "label": "QWERTY键盘",
                "bounds": [70, 580, 930, 990],
                "confidence": 0.95,
                "states": {"goal_relevant": False},
                "evidence": ["标准QWERTY按键可见"],
            },
        ]
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="empty-focused-input",
                    bounds=[150, 440, 850, 530],
                    text="",
                    placeholder="",
                    right_button=None,
                )
            ],
            keyboard={
                "visible": True,
                "bounds": [70, 580, 930, 990],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "mode_switch": None,
            },
        )
        provider = SequenceProvider([preliminary, audit])
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={"objective": "使唯一文本框最终显示小写文字 agent"},
        )

        candidate = scene.unique_trusted_goal_element()
        self.assertIsNotNone(candidate)
        self.assertEqual("local_audited_input_1", candidate.element_id)
        self.assertEqual("", candidate.states["value"])
        self.assertTrue(candidate.states["focused"])
        self.assertTrue(candidate.states["fully_visible"])
        self.assertEqual("qwerty", candidate.states["keyboard_layout"])
        self.assertEqual("direct_latin", candidate.states["keyboard_input_mode"])
        self.assertTrue(observer.last_diagnostics["input_structure_audit_used"])
        audit_prompt = provider.messages_seen[1][1]["content"][0]["text"]
        self.assertIn(
            '"input_mode":"unknown","case_mode":"unknown"',
            audit_prompt,
        )
        self.assertIn('"qwerty_anchors":', audit_prompt)
        self.assertIn('"backspace":[0,0]},"mode_switch":null', audit_prompt)
        self.assertIn("text-entry verification goal", audit_prompt)
        self.assertNotIn(
            '"input_mode":"chinese_pinyin","mode_switch":',
            audit_prompt,
        )

    def test_input_audit_rebinds_unfocused_input_without_full_visibility_evidence(self) -> None:
        preliminary = scene_payload()
        preliminary["elements"] = [
            {
                "element_id": "model-estimated-input",
                "role": "input",
                "meaning": "text_input_field",
                "label": "",
                "bounds": [120, 560, 880, 680],
                "confidence": 1.0,
                "states": {"value": "", "goal_relevant": True},
                "evidence": ["空矩形框"],
            }
        ]
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="complete-empty-input",
                    bounds=[130, 450, 870, 540],
                    text="",
                    placeholder="",
                    right_button=None,
                )
            ]
        )
        provider = SequenceProvider([preliminary, audit])
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={"objective": "当前输入框中的文字为 agent"},
        )

        candidate = scene.unique_trusted_goal_element()
        self.assertIsNotNone(candidate)
        self.assertEqual("local_audited_input_1", candidate.element_id)
        self.assertEqual((0.13, 0.45, 0.87, 0.54), candidate.bounds)
        self.assertTrue(candidate.states["fully_visible"])
        self.assertNotIn("focused", candidate.states)
        self.assertIs(candidate.states["soft_keyboard_visible"], False)
        self.assertIn(AUDITED_SOFT_KEYBOARD_HIDDEN_EVIDENCE, candidate.evidence)
        self.assertTrue(observer.last_diagnostics["input_structure_audit_used"])

    def test_empty_preliminary_overlays_do_not_mint_keyboard_hidden_evidence(self) -> None:
        preliminary = scene_payload()
        preliminary["overlays"] = []
        preliminary["elements"] = [
            {
                "element_id": "ordinary-button",
                "role": "button",
                "meaning": "open_details",
                "label": "查看详情",
                "bounds": [120, 420, 880, 540],
                "confidence": 1.0,
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": ["查看详情按钮"],
            }
        ]

        scene = GenericSceneObserver(SequenceProvider([preliminary])).observe(
            frames=stable_frames(),
            goal_context={"objective": "查看页面中的详情入口"},
        )

        self.assertFalse(
            any(
                AUDITED_SOFT_KEYBOARD_HIDDEN_EVIDENCE in element.evidence
                or element.states.get("soft_keyboard_visible") is False
                for element in scene.elements
            )
        )

    def test_input_audit_never_promotes_ime_preedit_region_to_application_input(self) -> None:
        empty = scene_payload()
        empty["elements"] = []
        audit = input_audit_payload(
            ime_preedit_regions=[
                {
                    "region_id": "composition-strip",
                    "bounds": [80, 420, 920, 500],
                    "text": "a'gen't",
                    "confidence": 0.98,
                }
            ],
            keyboard={
                "visible": True,
                "bounds": [0, 400, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "chinese_pinyin",
                "mode_switch": None,
            },
        )
        scene = GenericSceneObserver(
            SequenceProvider([empty, empty, audit])
        ).observe(
            frames=stable_frames(),
            goal_context={"objective": "读取当前应用输入框中的文字"},
        )

        self.assertFalse(any(item.role == "input" for item in scene.elements))
        self.assertIsNone(scene.unique_trusted_goal_element())

    def test_input_audit_rejects_application_claim_overlapping_ime_preedit(self) -> None:
        empty = scene_payload()
        empty["elements"] = []
        claimed_input = audited_application_input(
            structure_id="misclassified-preedit",
            bounds=[80, 420, 920, 500],
            text="a'gen't",
        )
        audit = input_audit_payload(
            application_inputs=[claimed_input],
            ime_preedit_regions=[
                {
                    "region_id": "same-region",
                    "bounds": [80, 420, 920, 500],
                    "text": "a'gen't",
                    "confidence": 0.98,
                }
            ],
            keyboard={
                "visible": True,
                "bounds": [0, 400, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "chinese_pinyin",
                "mode_switch": None,
            },
        )
        scene = GenericSceneObserver(
            SequenceProvider([empty, empty, audit])
        ).observe(
            frames=stable_frames(),
            goal_context={"objective": "读取当前应用输入框中的文字"},
        )

        self.assertFalse(any(item.role == "input" for item in scene.elements))

    def test_input_audit_with_no_trusted_structure_keeps_low_confidence_fail_closed(self) -> None:
        empty = scene_payload()
        empty["confidence"] = 0.6
        empty["elements"] = []
        provider = SequenceProvider([empty, empty, input_audit_payload()])

        with self.assertRaisesRegex(VisionAgentError, "整体置信度不足"):
            GenericSceneObserver(provider).observe(
                frames=stable_frames(),
                goal_context={"objective": "读取顶部空输入框"},
            )

        self.assertEqual(3, provider.calls)

    def test_empty_rectangle_without_literal_editable_cue_is_not_promoted(self) -> None:
        empty = scene_payload()
        empty["elements"] = []
        unproven = audited_application_input(text="", placeholder="")
        unproven["visible_editable_cues"] = []
        audit = input_audit_payload(application_inputs=[unproven])

        scene = GenericSceneObserver(
            SequenceProvider([empty, empty, audit])
        ).observe(
            frames=stable_frames(),
            goal_context={"objective": "读取顶部空输入框"},
        )

        self.assertFalse(any(item.role == "input" for item in scene.elements))

    def test_input_mode_goal_selects_only_compact_switch_inside_keyboard(self) -> None:
        empty = scene_payload()
        empty["elements"] = []
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    bounds=[80, 35, 820, 115],
                    text="",
                    placeholder="搜索",
                )
            ],
            keyboard={
                "visible": True,
                "bounds": [0, 360, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "chinese_pinyin",
                "mode_switch": {
                    "label": "中",
                    "bounds": [650, 900, 760, 970],
                    "confidence": 0.97,
                    "current_mode": "chinese_pinyin",
                    "target_mode": "direct_latin",
                },
            },
        )
        scene = GenericSceneObserver(
            SequenceProvider([empty, audit])
        ).observe(
            frames=stable_frames(),
            goal_context={"objective": "切换到英文直输模式 direct_latin"},
        )

        candidate = scene.unique_trusted_goal_element()
        self.assertIsNotNone(candidate)
        self.assertEqual("switch_keyboard_input_mode", candidate.meaning)
        input_element = scene.get_element("local_audited_input_1")
        self.assertFalse(input_element.states["goal_relevant"])

    def test_keyboard_mode_switch_conflicting_with_keyboard_mode_fails_closed(self) -> None:
        empty = scene_payload()
        empty["elements"] = []
        audit = input_audit_payload(
            keyboard={
                "visible": True,
                "bounds": [0, 360, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "mode_switch": {
                    "label": "中",
                    "bounds": [650, 900, 760, 970],
                    "confidence": 0.97,
                    "current_mode": "chinese_pinyin",
                    "target_mode": "direct_latin",
                },
            }
        )

        with self.assertRaisesRegex(VisionAgentError, "current_mode.*冲突"):
            GenericSceneObserver(
                SequenceProvider([empty, audit])
            ).observe(
                frames=stable_frames(),
                goal_context={"objective": "切换到英文直输模式"},
            )

    def test_chinese_pinyin_accepts_destination_english_mode_label(self) -> None:
        empty = scene_payload()
        empty["elements"] = []
        audit = input_audit_payload(
            ime_preedit_regions=[
                {
                    "region_id": "ime-preedit-1",
                    "bounds": [0, 600, 1000, 660],
                    "text": "longinp",
                    "confidence": 1.0,
                    "candidates": [
                        {
                            "text": "longinp",
                            "bounds": [60, 600, 260, 660],
                            "confidence": 1.0,
                            "fully_visible": True,
                        }
                    ],
                }
            ],
            keyboard={
                "visible": True,
                "bounds": [0, 660, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "chinese_pinyin",
                "mode_switch": {
                    "label": "英",
                    "bounds": [730, 930, 810, 980],
                    "confidence": 1.0,
                    "current_mode": "chinese_pinyin",
                    "target_mode": "direct_latin",
                },
            }
        )

        scene = GenericSceneObserver(
            SequenceProvider([empty, audit])
        ).observe(
            frames=stable_frames(),
            goal_context={"objective": "切换到英文直输模式 direct_latin"},
        )

        mode_switch = scene.get_element("local_audited_keyboard_mode_switch_1")
        self.assertEqual("switch_keyboard_input_mode", mode_switch.meaning)
        self.assertEqual("英", mode_switch.label)
        self.assertEqual("chinese_pinyin", mode_switch.states["current_mode"])
        self.assertEqual("direct_latin", mode_switch.states["target_mode"])

    def test_active_clear_binds_unique_ime_preedit_to_typed_input(self) -> None:
        base_scene = _parse_scene(
            json.dumps(scene_payload(), ensure_ascii=False),
            fingerprint="frame-clear-preedit",
        )
        context = {
            "entities": {
                "input_text": "freshsendproof",
                "active_subgoal_visual_context": {
                    "subgoal_id": "clear_draft",
                    "objective": "清除输入框中任何现有错误草稿且不要发送",
                    "constraints": ["不得发送任何现有错误草稿"],
                    "completion_conditions": ["输入框为空"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "input_text": "freshsendproof",
                        "active_input_transaction_text": "freshsendproof",
                        "active_input_field_id": "input_field_1",
                        "active_input_multiline": False,
                    },
                },
            }
        }

        def audit(*, preedits: list[dict], include_backspace: bool = True) -> dict:
            keyboard = {
                "visible": True,
                "bounds": [0, 660, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "chinese_pinyin",
                "case_mode": "lower",
                "qwerty_anchors": {
                    "q": [122, 710], "p": [880, 710],
                    "a": [164, 782], "l": [838, 782],
                    "z": [248, 853], "m": [754, 853],
                    "backspace": [880, 853],
                },
                "mode_switch": {
                    "label": "英",
                    "bounds": [730, 930, 810, 980],
                    "confidence": 1.0,
                    "current_mode": "chinese_pinyin",
                    "target_mode": "direct_latin",
                },
                "backspace_key": (
                    {
                        "label": "⌫",
                        "bounds": [830, 820, 940, 890],
                        "confidence": 1.0,
                        "fully_visible": True,
                    }
                    if include_backspace
                    else None
                ),
            }
            if not include_backspace:
                keyboard["qwerty_anchors"] = None
            return input_audit_payload(
                application_inputs=[
                    audited_application_input(
                        structure_id="message-field",
                        bounds=[130, 540, 700, 600],
                        text="",
                        visible_editable_cues=["cursor"],
                    )
                ],
                ime_preedit_regions=preedits,
                keyboard=keyboard,
            )

        unique_preedit = {
            "region_id": "ime-preedit-1",
            "bounds": [0, 600, 1000, 660],
            "text": "longinp",
            "confidence": 1.0,
            "candidates": [
                {
                    "text": text,
                    "bounds": [start, 605, start + 180, 655],
                    "confidence": 1.0,
                    "fully_visible": True,
                }
                for text, start in (("longinp", 20), ("longing", 220), ("Longines", 420))
            ],
        }
        projected = _apply_input_structure_audit(
            base_scene,
            json.dumps(audit(preedits=[unique_preedit]), ensure_ascii=False),
            fingerprint="frame-clear-preedit",
            goal_context=context,
            ledger_input_value="",
        )

        field = projected.get_element("local_audited_input_1")
        self.assertEqual("", field.states["value"])
        self.assertEqual("longinp", field.states["ime_preedit_text"])
        self.assertEqual("input_field_1", field.states["input_field_id"])
        self.assertTrue(field.states["goal_relevant"])
        self.assertEqual(
            "local_audited_input_1",
            projected.unique_trusted_goal_element().element_id,
        )

        ambiguous = _apply_input_structure_audit(
            base_scene,
            json.dumps(
                audit(
                    preedits=[
                        unique_preedit,
                        {
                            **unique_preedit,
                            "region_id": "ime-preedit-2",
                            "bounds": [0, 480, 1000, 530],
                            "candidates": [],
                        },
                    ]
                ),
                ensure_ascii=False,
            ),
            fingerprint="frame-clear-preedit",
            goal_context=context,
            ledger_input_value="",
        )
        self.assertNotIn(
            "ime_preedit_text",
            ambiguous.get_element("local_audited_input_1").states,
        )

        without_backspace = _apply_input_structure_audit(
            base_scene,
            json.dumps(
                audit(preedits=[unique_preedit], include_backspace=False),
                ensure_ascii=False,
            ),
            fingerprint="frame-clear-preedit",
            goal_context=context,
            ledger_input_value="",
        )
        self.assertNotIn(
            "ime_preedit_text",
            without_backspace.get_element("local_audited_input_1").states,
        )

        input_context = json.loads(json.dumps(context, ensure_ascii=False))
        active = input_context["entities"]["active_subgoal_visual_context"]
        active.update(
            {
                "subgoal_id": "input_text",
                "objective": "使用当前键盘输入精确正文 freshsendproof",
                "constraints": ["输入框为空"],
                "completion_conditions": ["输入框内容为 freshsendproof"],
            }
        )
        input_scene = _apply_input_structure_audit(
            base_scene,
            json.dumps(audit(preedits=[unique_preedit]), ensure_ascii=False),
            fingerprint="frame-clear-preedit",
            goal_context=input_context,
            ledger_input_value="",
        )
        input_field = input_scene.get_element("local_audited_input_1")
        self.assertEqual("longinp", input_field.states["ime_preedit_text"])
        self.assertTrue(input_field.states["goal_relevant"])
        self.assertEqual(
            "local_audited_input_1",
            input_scene.unique_trusted_goal_element().element_id,
        )

        for cue in ("longinp", "staledraft"):
            with self.subTest(inline_preedit_cue=cue):
                inline_audit = audit(preedits=[])
                inline_audit["application_inputs"][0][
                    "visible_editable_cues"
                ] = [cue]
                inline_scene = _apply_input_structure_audit(
                    base_scene,
                    json.dumps(inline_audit, ensure_ascii=False),
                    fingerprint="frame-inline-preedit",
                    goal_context=input_context,
                    ledger_input_value="",
                )
                inline_field = inline_scene.get_element("local_audited_input_1")
                self.assertEqual("", inline_field.states["value"])
                self.assertEqual(cue, inline_field.states["ime_preedit_text"])
                self.assertTrue(inline_field.states["goal_relevant"])

        for cues in (["cursor"], ["border"], ["longinp", "cursor"]):
            with self.subTest(non_authoritative_inline_cues=cues):
                rejected_audit = audit(preedits=[])
                rejected_audit["application_inputs"][0][
                    "visible_editable_cues"
                ] = cues
                rejected_scene = _apply_input_structure_audit(
                    base_scene,
                    json.dumps(rejected_audit, ensure_ascii=False),
                    fingerprint="frame-inline-preedit-rejected",
                    goal_context=input_context,
                    ledger_input_value="",
                )
                rejected_field = rejected_scene.get_element(
                    "local_audited_input_1"
                )
                self.assertNotIn("ime_preedit_text", rejected_field.states)

        nested_preedit = {
            "region_id": "ime-preedit-nested",
            "bounds": [155, 550, 275, 590],
            "text": "longinp",
            "confidence": 1.0,
            "candidates": [
                {
                    "text": text,
                    "bounds": [start, 605, start + 180, 655],
                    "confidence": 1.0,
                    "fully_visible": True,
                }
                for text, start in (
                    ("longinp", 20),
                    ("longing", 220),
                    ("Longines", 420),
                )
            ],
        }
        nested_audit = audit(preedits=[nested_preedit])
        nested_audit["application_inputs"][0]["visible_editable_cues"] = [
            "longinp",
            "caret",
        ]
        nested_scene = _apply_input_structure_audit(
            base_scene,
            json.dumps(nested_audit, ensure_ascii=False),
            fingerprint="frame-nested-preedit",
            goal_context=input_context,
            ledger_input_value="",
        )
        nested_field = nested_scene.get_element("local_audited_input_1")
        self.assertEqual("", nested_field.states["value"])
        self.assertEqual("longinp", nested_field.states["ime_preedit_text"])
        self.assertTrue(nested_field.states["goal_relevant"])

        keyboard_strip_audit = audit(preedits=[{
            "region_id": "ime-preedit-keyboard-strip",
            "bounds": [210, 290, 430, 330],
            "text": "freshsendproof",
            "confidence": 1.0,
            "candidates": [
                {
                    "text": text,
                    "bounds": [start, 590, start + width, 630],
                    "confidence": 1.0,
                    "fully_visible": True,
                }
                for text, start, width in (
                    ("freshsendproof", 110, 220),
                    ("fresh send proof", 350, 260),
                )
            ],
        }])
        keyboard_strip_audit["application_inputs"][0]["bounds"] = [
            140, 270, 850, 450,
        ]
        keyboard_strip_audit["application_inputs"][0][
            "visible_editable_cues"
        ] = ["freshsendproof"]
        keyboard_strip_audit["keyboard"]["bounds"] = [0, 570, 1000, 1000]
        keyboard_strip_audit["keyboard"]["qwerty_anchors"].update({
            "q": [70, 730], "p": [930, 730],
            "a": [140, 810], "l": [860, 810],
            "z": [280, 890], "m": [720, 890],
            "backspace": [930, 890],
        })
        keyboard_strip_scene = _apply_input_structure_audit(
            base_scene,
            json.dumps(keyboard_strip_audit, ensure_ascii=False),
            fingerprint="frame-keyboard-candidate-strip",
            goal_context=input_context,
            ledger_input_value="",
        )
        keyboard_candidate = keyboard_strip_scene.unique_trusted_goal_element()
        self.assertEqual("ime_exact_candidate", keyboard_candidate.meaning)
        self.assertEqual("freshsendproof", keyboard_candidate.label)

        split_candidate_rows = json.loads(
            json.dumps(keyboard_strip_audit, ensure_ascii=False)
        )
        split_candidate_rows["ime_preedit_regions"][0]["candidates"][1][
            "bounds"
        ] = [350, 650, 610, 690]
        with self.assertRaisesRegex(VisionAgentError, "同一水平候选行"):
            _apply_input_structure_audit(
                base_scene,
                json.dumps(split_candidate_rows, ensure_ascii=False),
                fingerprint="frame-split-keyboard-candidate-rows",
                goal_context=input_context,
                ledger_input_value="",
            )

        too_far = json.loads(json.dumps(nested_audit, ensure_ascii=False))
        too_far["ime_preedit_regions"][0]["candidates"][0]["bounds"] = [
            20, 760, 200, 810,
        ]
        with self.assertRaisesRegex(VisionAgentError, "紧邻候选行"):
            _apply_input_structure_audit(
                base_scene,
                json.dumps(too_far, ensure_ascii=False),
                fingerprint="frame-distant-candidate",
                goal_context=input_context,
                ledger_input_value="",
            )

        useful_direct_preedit = json.loads(
            json.dumps(nested_audit, ensure_ascii=False)
        )
        useful_region = useful_direct_preedit["ime_preedit_regions"][0]
        useful_region["text"] = "freshsendproof"
        useful_region["candidates"] = [
            {
                "text": "freshsendproof",
                "bounds": [20, 605, 260, 655],
                "confidence": 1.0,
                "fully_visible": True,
            },
            {
                "text": "fresh send proof",
                "bounds": [280, 605, 560, 655],
                "confidence": 1.0,
                "fully_visible": True,
            },
        ]
        useful_direct_preedit["application_inputs"][0][
            "visible_editable_cues"
        ] = ["freshsendproof"]
        useful_scene = _apply_input_structure_audit(
            base_scene,
            json.dumps(useful_direct_preedit, ensure_ascii=False),
            fingerprint="frame-useful-direct-preedit",
            goal_context=input_context,
            ledger_input_value="",
        )
        useful_field = useful_scene.get_element("local_audited_input_1")
        self.assertEqual("", useful_field.states["value"])
        self.assertEqual(
            "freshsendproof", useful_field.states["ime_preedit_text"]
        )
        self.assertEqual(
            "freshsendproof", useful_field.states["ime_exact_candidate_text"]
        )
        useful_candidate = useful_scene.unique_trusted_goal_element()
        self.assertEqual("ime_exact_candidate", useful_candidate.meaning)
        self.assertEqual("freshsendproof", useful_candidate.label)
        self.assertEqual(
            "freshsendproof",
            useful_candidate.states["expected_input_value"],
        )

        missing_useful_candidate = json.loads(
            json.dumps(useful_direct_preedit, ensure_ascii=False)
        )
        missing_useful_candidate["ime_preedit_regions"][0]["candidates"] = []
        with self.assertRaisesRegex(VisionAgentError, "不能转为清除"):
            _apply_input_structure_audit(
                base_scene,
                json.dumps(missing_useful_candidate, ensure_ascii=False),
                fingerprint="frame-useful-preedit-without-candidate",
                goal_context=input_context,
                ledger_input_value="",
            )

    def test_keyboard_mode_label_does_not_override_independent_direction(self) -> None:
        keyboard_bounds = (0.0, 360.0, 1000.0, 1000.0)
        for label, current_mode, target_mode in (
            ("中", "chinese_pinyin", "direct_latin"),
            ("中", "direct_latin", "chinese_pinyin"),
            ("EN", "chinese_pinyin", "direct_latin"),
            ("EN", "direct_latin", "chinese_pinyin"),
        ):
            with self.subTest(label=label, current_mode=current_mode):
                result = _validated_keyboard_mode_switch(
                    {
                        "label": label,
                        "bounds": [650, 900, 760, 970],
                        "confidence": 0.97,
                        "current_mode": current_mode,
                        "target_mode": target_mode,
                    },
                    keyboard_bounds=keyboard_bounds,
                )
                self.assertEqual(label, result["label"])
                self.assertEqual(current_mode, result["current_mode"])
                self.assertEqual(target_mode, result["target_mode"])

    def test_keyboard_mode_switch_keeps_direction_and_geometry_guards(self) -> None:
        keyboard_bounds = (0.0, 360.0, 1000.0, 1000.0)
        with self.assertRaisesRegex(UISceneError, "方向明确"):
            _validated_keyboard_mode_switch(
                {
                    "label": "英",
                    "bounds": [650, 900, 760, 970],
                    "confidence": 0.97,
                    "current_mode": "direct_latin",
                    "target_mode": "direct_latin",
                },
                keyboard_bounds=keyboard_bounds,
            )
        self.assertIsNone(
            _validated_keyboard_mode_switch(
                {
                    "label": "EN",
                    "bounds": [650, 900, 760, 970],
                    "confidence": 0.5,
                    "current_mode": "chinese_pinyin",
                    "target_mode": "direct_latin",
                },
                keyboard_bounds=keyboard_bounds,
            )
        )
        self.assertIsNone(
            _validated_keyboard_mode_switch(
                {
                    "label": "中",
                    "bounds": [50, 50, 150, 100],
                    "confidence": 0.97,
                    "current_mode": "direct_latin",
                    "target_mode": "chinese_pinyin",
                },
                keyboard_bounds=keyboard_bounds,
            )
        )

    def test_direct_latin_mode_accepts_matching_english_mode_label(self) -> None:
        empty = scene_payload()
        empty["elements"] = []
        audit = input_audit_payload(
            application_inputs=[audited_application_input(text="", placeholder="")],
            keyboard={
                "visible": True,
                "bounds": [0, 360, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "mode_switch": {
                    "label": "英",
                    "bounds": [650, 900, 760, 970],
                    "confidence": 0.97,
                    "current_mode": "direct_latin",
                    "target_mode": "chinese_pinyin",
                },
            },
        )
        audit["application_inputs"][0]["visible_editable_cues"] = ["caret"]

        scene = GenericSceneObserver(
            SequenceProvider([empty, empty, audit])
        ).observe(
            frames=stable_frames(),
            goal_context={"objective": "读取当前空白输入框和英文键盘"},
        )

        input_element = scene.get_element("local_audited_input_1")
        self.assertEqual("direct_latin", input_element.states["keyboard_input_mode"])
        mode_switch = scene.get_element("local_audited_keyboard_mode_switch_1")
        self.assertEqual("direct_latin", mode_switch.states["current_mode"])
        self.assertEqual("chinese_pinyin", mode_switch.states["target_mode"])

    def test_text_entry_discards_incomplete_non_target_mode_switch(self) -> None:
        empty = scene_payload()
        empty["elements"] = []
        audit = input_audit_payload(
            application_inputs=[audited_application_input(text="", placeholder="")],
            keyboard={
                "visible": True,
                "bounds": [0, 360, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "mode_switch": {
                    "label": "英",
                    "bounds": [650, 900, 760, 970],
                    "confidence": 0.97,
                    "current_mode": "direct_latin",
                },
            },
        )
        audit["application_inputs"][0]["visible_editable_cues"] = ["caret"]

        scene = GenericSceneObserver(
            SequenceProvider([empty, empty, audit])
        ).observe(
            frames=stable_frames(),
            goal_context={"objective": "让当前唯一空白输入框显示 agent，不提交"},
        )

        input_element = scene.unique_trusted_goal_element()
        self.assertIsNotNone(input_element)
        self.assertEqual("input", input_element.role)
        self.assertEqual("", input_element.states["value"])
        self.assertTrue(input_element.states["focused"])
        self.assertEqual("qwerty", input_element.states["keyboard_layout"])
        self.assertEqual("direct_latin", input_element.states["keyboard_input_mode"])
        self.assertFalse(
            any(
                item.element_id == "local_audited_keyboard_mode_switch_1"
                for item in scene.elements
            )
        )

    def test_text_entry_keeps_valid_non_target_mode_switch_non_goal_relevant(self) -> None:
        empty = scene_payload()
        empty["elements"] = []
        audit = input_audit_payload(
            application_inputs=[audited_application_input(text="", placeholder="正文")],
            keyboard={
                "visible": True,
                "bounds": [0, 360, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "mode_switch": {
                    "label": "中",
                    "bounds": [650, 900, 760, 970],
                    "confidence": 0.97,
                    "current_mode": "direct_latin",
                    "target_mode": "chinese_pinyin",
                },
            },
        )
        audit["application_inputs"][0]["visible_editable_cues"] = ["caret"]

        scene = GenericSceneObserver(
            SequenceProvider([empty, empty, audit])
        ).observe(
            frames=stable_frames(),
            goal_context={"objective": "让当前唯一输入框逐字显示 first，不提交"},
        )

        input_element = scene.unique_trusted_goal_element()
        self.assertIsNotNone(input_element)
        self.assertEqual("input", input_element.role)
        self.assertEqual("direct_latin", input_element.states["keyboard_input_mode"])
        mode_switch = scene.get_element("local_audited_keyboard_mode_switch_1")
        self.assertFalse(mode_switch.states["goal_relevant"])
        self.assertEqual("direct_latin", mode_switch.states["current_mode"])
        self.assertEqual("chinese_pinyin", mode_switch.states["target_mode"])

    def test_text_entry_result_discards_incomplete_non_target_mode_switch(self) -> None:
        empty = scene_payload()
        empty["elements"] = []
        audit = input_audit_payload(
            application_inputs=[audited_application_input(text="agent", placeholder="")],
            keyboard={
                "visible": True,
                "bounds": [0, 360, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "mode_switch": {
                    "label": "英",
                    "bounds": [650, 900, 760, 970],
                    "confidence": 0.97,
                    "current_mode": "direct_latin",
                },
            },
        )
        audit["application_inputs"][0]["visible_editable_cues"] = ["caret"]

        scene = GenericSceneObserver(
            SequenceProvider([empty, empty, audit])
        ).observe(
            frames=stable_frames(),
            goal_context={"objective": "让当前唯一输入框逐字显示 agent，不提交"},
        )

        input_element = scene.unique_trusted_goal_element()
        self.assertIsNotNone(input_element)
        self.assertEqual("agent", input_element.states["value"])
        self.assertEqual("direct_latin", input_element.states["keyboard_input_mode"])
        self.assertFalse(
            any(
                item.element_id == "local_audited_keyboard_mode_switch_1"
                for item in scene.elements
            )
        )

    def test_text_entry_result_discards_live_incomplete_switch_in_chinese_mode(self) -> None:
        empty = scene_payload()
        empty["elements"] = []
        audit = input_audit_payload(
            application_inputs=[audited_application_input(text="codex", placeholder="")],
            keyboard={
                "visible": True,
                "bounds": [0, 360, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "chinese_pinyin",
                "mode_switch": {
                    "label": "英",
                    "bounds": [650, 900, 760, 970],
                    "current_mode": "chinese_pinyin",
                },
            },
        )
        audit["application_inputs"][0]["visible_editable_cues"] = ["caret"]

        scene = GenericSceneObserver(
            SequenceProvider([empty, empty, audit])
        ).observe(
            frames=stable_frames(),
            goal_context={"objective": "核对当前唯一输入框逐字显示 codex，不提交"},
        )

        input_element = scene.unique_trusted_goal_element()
        self.assertIsNotNone(input_element)
        self.assertEqual("codex", input_element.states["value"])
        self.assertEqual("chinese_pinyin", input_element.states["keyboard_input_mode"])
        self.assertFalse(
            any(
                item.element_id == "local_audited_keyboard_mode_switch_1"
                for item in scene.elements
            )
        )

    def test_switch_goal_rejects_incomplete_mode_switch(self) -> None:
        empty = scene_payload()
        empty["elements"] = []
        audit = input_audit_payload(
            application_inputs=[audited_application_input(text="", placeholder="")],
            keyboard={
                "visible": True,
                "bounds": [0, 360, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "mode_switch": {
                    "label": "英",
                    "bounds": [650, 900, 760, 970],
                    "confidence": 0.97,
                    "current_mode": "direct_latin",
                },
            },
        )
        audit["application_inputs"][0]["visible_editable_cues"] = ["caret"]

        with self.assertRaisesRegex(VisionAgentError, "mode_switch.*字段"):
            GenericSceneObserver(
                SequenceProvider([empty, audit])
            ).observe(
                frames=stable_frames(),
                goal_context={"objective": "切换输入模式到中文拼音"},
            )

    def test_text_entry_rejects_mode_switch_with_extra_action_field(self) -> None:
        empty = scene_payload()
        empty["elements"] = []
        audit = input_audit_payload(
            application_inputs=[audited_application_input(text="", placeholder="")],
            keyboard={
                "visible": True,
                "bounds": [0, 360, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "mode_switch": {
                    "label": "英",
                    "bounds": [650, 900, 760, 970],
                    "confidence": 0.97,
                    "current_mode": "direct_latin",
                    "action": "tap",
                },
            },
        )
        audit["application_inputs"][0]["visible_editable_cues"] = ["caret"]

        observer = GenericSceneObserver(SequenceProvider([empty, empty, audit]))
        with self.assertRaisesRegex(VisionAgentError, "mode_switch.*字段"):
            observer.observe(
                frames=stable_frames(),
                goal_context={"objective": "让当前唯一空白输入框显示 agent，不提交"},
            )
        self.assertEqual(
            ["action", "bounds", "confidence", "current_mode", "label"],
            observer.status()["last_input_structure_shape"]["mode_switch_keys"],
        )
        self.assertNotIn(
            "agent",
            json.dumps(
                observer.status()["last_input_structure_shape"],
                ensure_ascii=False,
            ),
        )

    def test_input_structure_shape_never_retains_observed_text(self) -> None:
        audit = input_audit_payload(
            application_inputs=[audited_application_input(text="privatevalue")],
        )

        shape = _input_structure_diagnostic_shape(
            json.dumps(audit, ensure_ascii=False)
        )

        self.assertTrue(shape["parseable"])
        self.assertEqual("NoneType", shape["mode_switch_type"])
        self.assertNotIn("privatevalue", json.dumps(shape, ensure_ascii=False))

    def test_non_input_goal_discards_malformed_peripheral_keyboard_switch(self) -> None:
        payload = scene_payload()
        payload["elements"] = [
            {
                "element_id": "refresh",
                "role": "button",
                "meaning": "refresh",
                "label": "刷新",
                "bounds": [850, 20, 920, 90],
                "confidence": 0.98,
                "states": {"goal_relevant": True},
                "evidence": ["当前页面顶部刷新控件"],
            },
            {
                "element_id": "bad-switch",
                "role": "button",
                "meaning": "switch_keyboard_input_mode",
                "label": "英",
                "bounds": [690, 890, 790, 930],
                "confidence": 0.92,
                "states": {
                    "goal_relevant": False,
                    "keyboard_input_mode_switch": True,
                    "current_mode": "chinese_pinyin",
                    "target_mode": "direct_latin",
                },
                "evidence": ["方向与键面文字冲突"],
            },
        ]

        provider = SequenceProvider(
            [
                payload,
                icon_cluster_audit_payload(),
                localized_icon_cluster_audit_payload(),
            ]
        )
        scene = GenericSceneObserver(provider).observe(
            frames=icon_cluster_frames(),
            goal_context={"objective": "当前页面完成一次重新加载"},
        )

        self.assertEqual(3, provider.calls)
        self.assertEqual(
            "local_audited_reload_control_1",
            scene.unique_trusted_goal_element().element_id,
        )
        self.assertTrue(
            scene.unique_trusted_goal_element().states["reload_visual_audit"]
        )

    def test_attested_reload_survives_rejected_unconsumed_input_audit(self) -> None:
        compact = scene_payload()
        compact["elements"] = [
            {
                "element_id": "model-reload",
                "role": "icon",
                "meaning": "reload",
                "label": "",
                "bounds": [820, 20, 875, 75],
                "confidence": 0.96,
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": ["右上方完整圆形箭头"],
            }
        ]
        invalid_input_audit = input_audit_payload(
            application_inputs=[audited_application_input(text="agent")],
            keyboard={
                "visible": True,
                "bounds": [0, 360, 1000, 1210],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "mode_switch": None,
            },
        )
        provider = SequenceProvider(
            [
                compact,
                icon_cluster_audit_payload(),
                localized_icon_cluster_audit_payload(),
                invalid_input_audit,
            ]
        )
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=icon_cluster_frames(),
            goal_context={
                "objective": "重新加载当前页面，使唯一输入框恢复为空",
            },
        )

        candidate = scene.unique_trusted_goal_element()
        self.assertEqual(4, provider.calls)
        self.assertEqual("local_audited_reload_control_1", candidate.element_id)
        self.assertTrue(candidate.states["reload_visual_audit"])
        self.assertTrue(
            observer.last_diagnostics[
                "input_structure_audit_isolated_from_attested_non_input"
            ]
        )

    def test_active_reload_focus_does_not_run_later_input_subgoal_audit(self) -> None:
        compact = scene_payload()
        compact["elements"] = [
            {
                "element_id": "model-reload",
                "role": "icon",
                "meaning": "reload",
                "label": "",
                "bounds": [820, 20, 875, 75],
                "confidence": 0.96,
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": ["右上方完整圆形箭头"],
            }
        ]
        provider = SequenceProvider(
            [
                compact,
                icon_cluster_audit_payload(),
                localized_icon_cluster_audit_payload(),
            ]
        )
        observer = GenericSceneObserver(provider)
        context = {
            "objective": "先重新加载页面，再确认输入框为空",
            "entities": {
                "original_goal_visual_context": "刷新后让输入框恢复为空",
                "active_subgoal_visual_context": {
                    "subgoal_id": "reload_page",
                    "objective": "重新加载当前页面",
                    "constraints": ["软键盘必须保持不可见"],
                    "completion_conditions": ["页面内容已重新加载"],
                    "execution_class": "navigate",
                    "goal_entities": {"input_text": "agent"},
                },
            },
        }

        scene = observer.observe(frames=icon_cluster_frames(), goal_context=context)

        self.assertEqual(3, provider.calls)
        self.assertEqual(
            "local_audited_reload_control_1",
            scene.unique_trusted_goal_element().element_id,
        )
        self.assertFalse(observer.last_diagnostics["input_structure_audit_used"])

    def test_active_hide_keyboard_focus_still_requests_input_audit(self) -> None:
        context = {
            "objective": "重新加载后输入 agent 并隐藏键盘",
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "hide_keyboard",
                    "objective": "让当前软键盘保持不可见",
                    "constraints": [],
                    "completion_conditions": ["软键盘未显示"],
                    "execution_class": "navigate",
                    "goal_entities": {"input_text": "agent"},
                }
            },
        }

        self.assertTrue(_goal_requests_input(context))

    def test_temporary_draft_area_is_a_generic_input_audit_goal(self) -> None:
        context = {
            "objective": "恢复当前临时草稿区域为空白",
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "clear_draft",
                    "objective": "当前页面唯一临时草稿区域内容为空白",
                    "constraints": [],
                    "completion_conditions": [
                        "草稿区域显示为空白",
                        "键盘仍然可见",
                    ],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "target_ui_label": "唯一临时草稿区域"
                    },
                }
            },
        }

        self.assertTrue(_goal_requests_input(context))

    def test_hide_keyboard_allows_boundsless_presence_but_not_input_mode(self) -> None:
        compact = scene_payload()
        compact["summary"] = "唯一输入框为agent，当前软键盘可见。"
        compact["overlays"] = ["软键盘"]
        compact["elements"] = [
            {
                "element_id": "model-input",
                "role": "input",
                "meaning": "application_text_input",
                "label": "agent",
                "bounds": [120, 360, 880, 470],
                "confidence": 0.98,
                "states": {
                    "goal_relevant": True,
                    "fully_visible": False,
                    "value": "agent",
                },
                "evidence": ["输入框边框和光标可见"],
            }
        ]
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    text="agent",
                    placeholder="",
                )
            ],
            keyboard={
                "visible": True,
                "bounds": [0, 580, 1000, 1210],
                "layout": "qwerty",
                "input_mode": "chinese_pinyin",
                "qwerty_anchors": {
                    "q": [115, 704],
                    "p": [875, 704],
                    "a": [157, 773],
                    "l": [832, 773],
                    "z": [241, 844],
                    "m": [747, 844],
                    "backspace": [875, 844],
                },
                "mode_switch": {
                    "label": "英",
                    "bounds": [760, 1080, 850, 1160],
                    "current_mode": "chinese_pinyin",
                },
            },
        )
        context = {
            "objective": "输入完成后让软键盘不可见",
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "hide_keyboard",
                    "objective": "使软键盘最终不在画面中",
                    "constraints": ["保持输入框内容不变"],
                    "completion_conditions": ["软键盘不可见"],
                    "execution_class": "navigate",
                    "goal_entities": {"input_text": "agent"},
                }
            },
        }

        scene = GenericSceneObserver(
            SequenceProvider([compact, audit])
        ).observe(
            frames=stable_frames(),
            goal_context=context,
        )

        target = scene.unique_trusted_goal_element()
        self.assertEqual("local_audited_input_1", target.element_id)
        self.assertTrue(target.states["focused"])
        self.assertEqual("unknown", target.states["keyboard_layout"])
        self.assertEqual("unknown", target.states["keyboard_input_mode"])
        self.assertFalse(
            any(
                item.meaning == "switch_keyboard_input_mode"
                for item in scene.elements
            )
        )

    def test_hidden_keyboard_fact_survives_rejected_input_geometry_without_authority(self) -> None:
        compact = scene_payload()
        compact["summary"] = "输入框中显示codex，软键盘已收起。"
        compact["overlays"] = []
        compact["elements"] = [
            {
                "element_id": "model-input",
                "role": "input",
                "meaning": "application_text_input",
                "label": "codex",
                "bounds": [120, 820, 880, 900],
                "confidence": 0.98,
                "states": {
                    "goal_relevant": True,
                    "fully_visible": True,
                    "value": "codex",
                    "focused": True,
                },
                "evidence": ["输入框内逐字显示codex"],
            }
        ]
        audit = input_audit_payload(
            application_inputs=[
                {
                    "structure_id": "app-input-1",
                    "bounds": [130, 1590, 850, 1690],
                    "fully_visible": True,
                    "text": "codex",
                    "placeholder": "",
                    "visible_editable_cues": ["caret"],
                    "confidence": 1.0,
                    "right_button": None,
                }
            ],
            keyboard={
                "visible": False,
                "bounds": None,
                "layout": "unknown",
                "input_mode": "unknown",
                "qwerty_anchors": None,
                "mode_switch": None,
            },
        )
        context = {
            "objective": "输入后收起软键盘",
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "hide_keyboard",
                    "objective": "软键盘已收起",
                    "constraints": ["保持输入框内容不变"],
                    "completion_conditions": ["软键盘不可见"],
                    "execution_class": "navigate",
                    "goal_entities": {"input_text": "codex"},
                }
            },
        }

        scene = GenericSceneObserver(SequenceProvider([compact, audit])).observe(
            frames=stable_frames(),
            goal_context=context,
        )

        self.assertIn("输入结构只读审计确认应用输入框当前文字：codex", scene.summary)
        self.assertIn(AUDITED_SOFT_KEYBOARD_HIDDEN_EVIDENCE, scene.summary)
        self.assertTrue(
            all(item.states.get("goal_relevant") is False for item in scene.elements)
        )
        self.assertIsNone(scene.unique_trusted_goal_element())

    def test_input_audit_mints_only_unique_exact_chinese_candidate(self) -> None:
        base = scene_payload()
        base_scene = _parse_scene(
            json.dumps(base, ensure_ascii=False),
            fingerprint="frame-ime",
        )
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="message-field",
                    bounds=[80, 120, 920, 210],
                    text="",
                )
            ],
            ime_preedit_regions=[
                {
                    "region_id": "candidate-strip",
                    "bounds": [40, 380, 960, 470],
                    "text": "ni'hao",
                    "confidence": 0.98,
                    "candidates": [
                        {
                            "text": "你好",
                            "bounds": [80, 392, 220, 458],
                            "confidence": 0.98,
                            "fully_visible": True,
                        },
                        {
                            "text": "拟好",
                            "bounds": [250, 392, 390, 458],
                            "confidence": 0.96,
                            "fully_visible": True,
                        },
                    ],
                }
            ],
            keyboard={
                "visible": True,
                "bounds": [0, 480, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "chinese_pinyin",
                "qwerty_anchors": {
                    "q": [115, 610], "p": [875, 610],
                    "a": [157, 700], "l": [832, 700],
                    "z": [241, 790], "m": [747, 790],
                    "backspace": [875, 790],
                },
                "mode_switch": None,
            },
        )

        scene = _apply_input_structure_audit(
            base_scene,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="frame-ime",
            goal_context={
                "objective": "输入你好但不要发送",
                "entities": {"input_text": "你好"},
            },
        )

        target = scene.unique_trusted_goal_element()
        self.assertEqual("local_audited_ime_candidate_1", target.element_id)
        self.assertEqual("你好", target.label)
        field = scene.get_element("local_audited_input_1")
        self.assertEqual("nihao", field.states["ime_preedit_text"])
        self.assertEqual("你好", field.states["ime_exact_candidate_text"])
        self.assertEqual("nihao", target.states["pinyin"])

        audit["ime_preedit_regions"][0]["text"] = "ni’hao"
        separator_variation = _apply_input_structure_audit(
            base_scene,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="frame-ime-curly-separator",
            goal_context={
                "objective": "输入你好但不要发送",
                "entities": {"input_text": "你好"},
            },
        )
        variation_field = separator_variation.get_element(
            "local_audited_input_1"
        )
        variation_candidate = separator_variation.get_element(
            "local_audited_ime_candidate_1"
        )
        self.assertEqual("nihao", variation_field.states["ime_preedit_text"])
        self.assertEqual("nihao", variation_candidate.states["pinyin"])

    def test_qwerty_secondary_digit_hint_is_not_a_direct_literal_key(self) -> None:
        base_scene = _parse_scene(
            json.dumps(scene_payload(), ensure_ascii=False),
            fingerprint="frame-secondary-digit",
        )

        def audit(*, literal_bounds, include_numeric_switch):
            return input_audit_payload(
                application_inputs=[
                    audited_application_input(
                        structure_id="message-field",
                        bounds=[150, 530, 700, 590],
                        text="longinput",
                    )
                ],
                keyboard={
                    "visible": True,
                    "bounds": [0, 580, 1000, 1000],
                    "layout": "qwerty",
                    "input_mode": "direct_latin",
                    "case_mode": "lower",
                    "qwerty_anchors": {
                        "q": [80, 730], "p": [910, 730],
                        "a": [130, 810], "l": [820, 810],
                        "z": [240, 890], "m": [750, 890],
                        "backspace": [880, 890],
                    },
                    "mode_switch": None,
                    "backspace_key": None,
                    "case_switch": None,
                    "literal_keys": [
                        {
                            "value": "2", "label": "2",
                            "key_kind": "character",
                            "bounds": literal_bounds,
                            "confidence": 1.0,
                            "fully_visible": True,
                        }
                    ],
                    "layout_switches": (
                        [
                            {
                                "label": "123",
                                "bounds": [20, 900, 180, 980],
                                "confidence": 1.0,
                                "current_layout": "qwerty",
                                "target_layout": "numeric",
                            }
                        ]
                        if include_numeric_switch
                        else []
                    ),
                },
            )

        hint_scene = _apply_input_structure_audit(
            base_scene,
            json.dumps(
                audit(
                    literal_bounds=[170, 690, 250, 750],
                    include_numeric_switch=True,
                ),
                ensure_ascii=False,
            ),
            fingerprint="frame-secondary-digit",
            goal_context={
                "objective": "输入框逐字等于目标且不发送",
                "entities": {
                    "input_text": "longinput2026abcdefghijklmnopqrstuvwxyz"
                },
            },
        )
        target = hint_scene.unique_trusted_goal_element()
        self.assertEqual(
            "local_audited_keyboard_layout_switch_1",
            target.element_id,
        )
        self.assertEqual("numeric", target.states["target_layout"])
        self.assertNotIn(
            "local_audited_literal_key_1",
            {item.element_id for item in hint_scene.elements},
        )

        no_switch_scene = _apply_input_structure_audit(
            base_scene,
            json.dumps(
                audit(
                    literal_bounds=[170, 690, 250, 750],
                    include_numeric_switch=False,
                ),
                ensure_ascii=False,
            ),
            fingerprint="frame-secondary-digit-no-switch",
            goal_context={
                "objective": "输入框逐字等于目标且不发送",
                "entities": {
                    "input_text": "longinput2026abcdefghijklmnopqrstuvwxyz"
                },
            },
        )
        self.assertIsNone(no_switch_scene.unique_trusted_goal_element())

        dedicated_row_scene = _apply_input_structure_audit(
            base_scene,
            json.dumps(
                audit(
                    literal_bounds=[170, 590, 250, 650],
                    include_numeric_switch=False,
                ),
                ensure_ascii=False,
            ),
            fingerprint="frame-dedicated-digit-row",
            goal_context={
                "objective": "输入框逐字等于目标且不发送",
                "entities": {
                    "input_text": "longinput2026abcdefghijklmnopqrstuvwxyz"
                },
            },
        )
        dedicated = dedicated_row_scene.unique_trusted_goal_element()
        self.assertEqual("local_audited_literal_key_1", dedicated.element_id)
        self.assertEqual("2", dedicated.states["key_value"])

    def test_numeric_live_audit_keeps_generic_backspace_geometry(self) -> None:
        base_scene = _parse_scene(
            json.dumps(scene_payload(), ensure_ascii=False),
            fingerprint="frame-live-numeric",
        )
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="message-field",
                    bounds=[150, 530, 700, 590],
                    text="live",
                )
            ],
            keyboard={
                "visible": True,
                "bounds": [0, 660, 1000, 1000],
                "layout": "numeric",
                "input_mode": "chinese_pinyin",
                "case_mode": "unknown",
                "qwerty_anchors": None,
                "mode_switch": None,
                "backspace_key": {
                    "label": "⌫",
                    "bounds": [820, 680, 980, 740],
                    "confidence": 1.0,
                    "fully_visible": True,
                },
                "case_switch": None,
                "literal_keys": [
                    {
                        "value": "2", "label": "2", "key_kind": "character",
                        "bounds": [420, 680, 580, 740], "confidence": 1.0,
                        "fully_visible": True,
                    },
                ],
                "layout_switches": [
                    {
                        "label": "返回", "bounds": [220, 880, 380, 940],
                        "confidence": 1.0, "current_layout": "numeric",
                        "target_layout": "qwerty",
                    }
                ],
            },
        )
        current = _apply_input_structure_audit(
            base_scene,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="frame-live-numeric",
            goal_context={
                "objective": "输入框最终逐字显示 live21 且不发送",
                "entities": {"input_text": "live21"},
            },
        )

        target = current.unique_trusted_goal_element()
        self.assertEqual("local_audited_literal_key_1", target.element_id)
        self.assertEqual("2", target.states["key_value"])
        self.assertEqual("live2", target.states["expected_input_value"])
        field = current.get_element("local_audited_input_1")
        self.assertEqual("numeric", field.states["keyboard_layout"])
        self.assertEqual(
            {
                "type": "generic",
                "anchors": {"backspace": [900.0, 710.0]},
                "source": "input_structure_audit",
            },
            field.states["keyboard_geometry"],
        )

    def test_input_audit_parser_uses_exact_observed_next_literal_beyond_prefix_cap(
        self,
    ) -> None:
        target = "复杂输入验收2026:123+45-6@7."
        for current, expected in (
            ("复杂输入验收2026:123+4", "5"),
            ("复杂输入验收2026:123+45", "-"),
        ):
            with self.subTest(current=current, expected=expected):
                base_scene = _parse_scene(
                    json.dumps(scene_payload(), ensure_ascii=False),
                    fingerprint="frame-literal-parser",
                )
                audit = input_audit_payload(
                    application_inputs=[
                        audited_application_input(
                            structure_id="message-field",
                            bounds=[150, 530, 700, 590],
                            text=current,
                        )
                    ],
                    keyboard={
                        "visible": True,
                        "bounds": [0, 640, 1000, 1000],
                        "layout": "numeric",
                        "input_mode": "direct_latin",
                        "case_mode": "unknown",
                        "qwerty_anchors": None,
                        "mode_switch": None,
                        "backspace_key": None,
                        "case_switch": None,
                        "literal_keys": [
                            {
                                "value": expected,
                                "label": expected,
                                "key_kind": "character",
                                "bounds": [420, 750, 580, 810],
                                "confidence": 1.0,
                                "fully_visible": True,
                            }
                        ],
                        "layout_switches": [],
                    },
                )
                scene = _apply_input_structure_audit(
                    base_scene,
                    json.dumps(audit, ensure_ascii=False),
                    fingerprint="frame-literal-parser",
                    goal_context={
                        "objective": "输入框逐字等于目标且不发送",
                        "entities": {"input_text": target},
                    },
                )
                key = scene.unique_trusted_goal_element()
                self.assertEqual(expected, key.states["key_value"])
                self.assertEqual(current + expected, key.states["expected_input_value"])

                wrong = json.loads(json.dumps(audit, ensure_ascii=False))
                wrong["keyboard"]["literal_keys"][0].update(
                    {"value": "6", "label": "6"}
                )
                projected = _apply_input_structure_audit(
                    base_scene,
                    json.dumps(wrong, ensure_ascii=False),
                    fingerprint="frame-literal-parser-wrong",
                    goal_context={
                        "objective": "输入框逐字等于目标且不发送",
                        "entities": {"input_text": target},
                    },
                )
                self.assertFalse(
                    any(
                        item.states.get("key_value") == "6"
                        for item in projected.elements
                    )
                )

    def test_input_audit_discards_literal_key_outside_goal_whitelist(self) -> None:
        base_scene = _parse_scene(
            json.dumps(scene_payload(), ensure_ascii=False),
            fingerprint="frame-literal-whitelist",
        )
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="search-field",
                    bounds=[80, 120, 920, 210],
                    text="",
                )
            ],
            keyboard={
                "visible": True,
                "bounds": [0, 480, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "case_mode": "lower",
                "qwerty_anchors": {
                    "q": [115, 610], "p": [875, 610],
                    "a": [157, 700], "l": [832, 700],
                    "z": [241, 790], "m": [747, 790],
                    "backspace": [875, 790],
                },
                "mode_switch": None,
                "case_switch": None,
                "literal_keys": [
                    {
                        "value": "w", "label": "w", "key_kind": "character",
                        "bounds": [170, 580, 240, 660], "confidence": 0.98,
                        "fully_visible": True,
                    }
                ],
                "layout_switches": [],
            },
        )

        projected = _apply_input_structure_audit(
            base_scene,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="frame-literal-whitelist",
            goal_context={
                "objective": "输入 wifi",
                "entities": {"input_text": "wifi"},
            },
        )
        self.assertFalse(
            any(item.states.get("key_value") == "w" for item in projected.elements)
        )

    def test_input_audit_projects_optional_extras_and_keeps_exact_layout_switch(
        self,
    ) -> None:
        current_value = "longinputvalidation2026"
        target_value = current_value + ":123+45-6@7."
        payload = scene_payload()
        payload["elements"] = [
            {
                "element_id": "coarse-input",
                "role": "input",
                "meaning": "application_text_input",
                "label": current_value,
                "bounds": [120, 530, 700, 590],
                "confidence": 1.0,
                "states": {
                    "goal_relevant": True,
                    "fully_visible": True,
                    "value": current_value,
                    "focused": True,
                },
                "evidence": [current_value],
            }
        ]
        base_scene = _parse_scene(
            json.dumps(payload, ensure_ascii=False),
            fingerprint="frame-optional-input-extras",
        )
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="app-input-1",
                    bounds=[130, 540, 700, 590],
                    text=current_value,
                )
            ],
            ime_preedit_regions=[
                {
                    "region_id": "ime-preedit-1",
                    "bounds": [80, 600, 920, 660],
                    "text": current_value,
                    "confidence": 1.0,
                    "candidates": [],
                }
            ],
            keyboard={
                "visible": True,
                "bounds": [0, 660, 1000, 1000],
                "layout": "numeric",
                "input_mode": "direct_latin",
                "case_mode": "unknown",
                "qwerty_anchors": None,
                "mode_switch": None,
                "backspace_key": None,
                "case_switch": None,
                "literal_keys": [
                    {
                        "value": value,
                        "label": value,
                        "key_kind": "character",
                        "bounds": bounds,
                        "confidence": 1.0,
                        "fully_visible": True,
                    }
                    for value, bounds in (
                        ("+", [20, 790, 180, 850]),
                        ("@", [820, 790, 980, 850]),
                    )
                ],
                "layout_switches": [
                    {
                        "label": "!?#",
                        "bounds": [20, 850, 180, 910],
                        "confidence": 1.0,
                        "current_layout": "numeric",
                        "target_layout": "symbol",
                    }
                ],
            },
        )

        projected = _apply_input_structure_audit(
            base_scene,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="frame-optional-input-extras",
            goal_context={
                "objective": "输入框逐字等于目标且不发送",
                "entities": {"input_text": target_value},
            },
            ledger_input_value=current_value,
        )

        target = projected.unique_trusted_goal_element()
        self.assertEqual("local_audited_keyboard_layout_switch_1", target.element_id)
        self.assertEqual("symbol", target.states["target_layout"])
        self.assertFalse(
            any(
                item.states.get("key_value") in {"+", "@"}
                for item in projected.elements
            )
        )

    def test_keyboard_layout_switch_uses_only_unique_monotonic_next_hop(
        self,
    ) -> None:
        qwerty_to_numeric = {
            "label": "123",
            "bounds": [100, 900, 200, 980],
            "confidence": 1.0,
            "current_layout": "qwerty",
            "target_layout": "numeric",
        }
        qwerty_to_symbol = {
            "label": "符",
            "bounds": [220, 900, 320, 980],
            "confidence": 1.0,
            "current_layout": "qwerty",
            "target_layout": "symbol",
        }

        intermediate = _select_keyboard_layout_switch_for_target(
            [qwerty_to_numeric],
            current_layout="qwerty",
            target_layout="symbol",
        )
        self.assertEqual("numeric", intermediate["target_layout"])

        direct = _select_keyboard_layout_switch_for_target(
            [qwerty_to_numeric, qwerty_to_symbol],
            current_layout="qwerty",
            target_layout="symbol",
        )
        self.assertEqual("symbol", direct["target_layout"])

        self.assertIsNone(
            _select_keyboard_layout_switch_for_target(
                [qwerty_to_numeric, {**qwerty_to_numeric, "label": "数字"}],
                current_layout="qwerty",
                target_layout="symbol",
            )
        )

    def test_fullwidth_symbol_input_projects_numeric_intermediate_layout_hop(
        self,
    ) -> None:
        current_value = "aaazjie"
        payload = scene_payload()
        payload["elements"] = []
        base_scene = _parse_scene(
            json.dumps(payload, ensure_ascii=False),
            fingerprint="frame-symbol-intermediate-hop",
        )
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="app-input-1",
                    bounds=[130, 570, 720, 630],
                    text=current_value,
                    visible_editable_cues=[current_value, "|"],
                    caret_line_index=0,
                )
            ],
            keyboard={
                "visible": True,
                "bounds": [0, 640, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "case_mode": "lower",
                "qwerty_anchors": {
                    "q": [80, 730], "p": [920, 730],
                    "a": [130, 820], "l": [870, 820],
                    "z": [230, 900], "m": [770, 900],
                    "backspace": [920, 900],
                },
                "mode_switch": None,
                "backspace_key": None,
                "enter_key": None,
                "case_switch": None,
                "literal_keys": [
                    {
                        "value": "？",
                        "label": "?",
                        "key_kind": "character",
                        "bounds": [820, 820, 880, 860],
                        "confidence": 0.9,
                        "fully_visible": True,
                    }
                ],
                "layout_switches": [
                    {
                        "label": "123",
                        "bounds": [180, 940, 280, 980],
                        "confidence": 1.0,
                        "current_layout": "qwerty",
                        "target_layout": "numeric",
                    }
                ],
            },
        )

        projected = _apply_input_structure_audit(
            base_scene,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="frame-symbol-intermediate-hop",
            goal_context={
                "objective": "输入框逐字等于授权文字且不发送",
                "entities": {
                    "input_text": current_value + "？你好",
                    "active_input_transaction_text": current_value + "？你好",
                    "active_input_field_id": "input_field_1",
                },
            },
            ledger_input_value=current_value,
        )

        target = projected.unique_trusted_goal_element()
        self.assertEqual(
            "local_audited_keyboard_layout_switch_1",
            target.element_id,
        )
        self.assertEqual("qwerty", target.states["current_layout"])
        self.assertEqual("numeric", target.states["target_layout"])
        self.assertEqual(current_value, target.states["prior_input_value"])

    def test_input_audit_rejects_compact_direct_latin_value_shadow(
        self,
    ) -> None:
        current_value = "longinputvalidation2026"
        target_value = current_value + ":123+45-6@7."
        payload = scene_payload()
        payload["elements"] = [
            {
                "element_id": "coarse-input",
                "role": "input",
                "meaning": "application_text_input",
                "label": current_value,
                "bounds": [120, 530, 700, 590],
                "confidence": 1.0,
                "states": {
                    "goal_relevant": True,
                    "fully_visible": True,
                    "value": current_value,
                    "focused": True,
                },
                "evidence": [current_value],
            }
        ]
        base_scene = _parse_scene(
            json.dumps(payload, ensure_ascii=False),
            fingerprint="frame-same-value-preedit",
        )

        def parsed(*, input_mode: str, preedit_text: str, coarse_value: str):
            audit = input_audit_payload(
                application_inputs=[
                    audited_application_input(
                        structure_id="app-input-1",
                        bounds=[140, 530, 710, 590],
                        text="",
                    )
                ],
                ime_preedit_regions=[
                    {
                        "region_id": "ime-preedit-1",
                        "bounds": [100, 600, 900, 660],
                        "text": preedit_text,
                        "confidence": 1.0,
                        "candidates": [
                            {
                                "text": preedit_text,
                                "bounds": [100, 600, 850, 660],
                                "confidence": 1.0,
                                "fully_visible": True,
                            }
                        ],
                    }
                ],
                keyboard={
                    "visible": True,
                    "bounds": [0, 660, 1000, 1000],
                    "layout": "numeric",
                    "input_mode": input_mode,
                    "case_mode": "unknown",
                    "qwerty_anchors": None,
                    "mode_switch": None,
                    "backspace_key": None,
                    "case_switch": None,
                    "literal_keys": [],
                    "layout_switches": [
                        {
                            "label": "!?#",
                            "bounds": [10, 890, 190, 950],
                            "confidence": 1.0,
                            "current_layout": "numeric",
                            "target_layout": "symbol",
                        }
                    ],
                },
            )
            return _apply_input_structure_audit(
                base_scene,
                json.dumps(audit, ensure_ascii=False),
                fingerprint="frame-same-value-preedit",
                goal_context={
                    "objective": "输入框逐字等于目标且不发送",
                    "entities": {"input_text": target_value},
                },
                ledger_input_value=coarse_value,
            )

        rejected_shadow = parsed(
            input_mode="direct_latin",
            preedit_text=current_value,
            coarse_value=current_value,
        )
        field = rejected_shadow.get_element("local_audited_input_1")
        self.assertEqual("", field.states["value"])

        for input_mode, preedit_text, coarse_value in (
            ("direct_latin", "different-visible-value", current_value),
            ("direct_latin", current_value, "different-coarse-value"),
            ("chinese_pinyin", current_value, current_value),
        ):
            with self.subTest(
                input_mode=input_mode,
                preedit_text=preedit_text,
                coarse_value=coarse_value,
            ):
                rejected = parsed(
                    input_mode=input_mode,
                    preedit_text=preedit_text,
                    coarse_value=coarse_value,
                )
                rejected_field = rejected.get_element("local_audited_input_1")
                self.assertEqual("", rejected_field.states["value"])
                self.assertNotIn(
                    "same_frame_visible_cue_text",
                    rejected_field.states,
                )

    def test_input_audit_rejects_compact_exact_cue_shadow_without_lineage(
        self,
    ) -> None:
        current_value = "freshsendproof"
        payload = scene_payload()
        payload["elements"] = [
            {
                "element_id": "coarse-input",
                "role": "input",
                "meaning": "application_text_input",
                "label": current_value,
                "bounds": [120, 530, 700, 590],
                "confidence": 1.0,
                "states": {
                    "goal_relevant": True,
                    "fully_visible": True,
                    "value": current_value,
                    "focused": True,
                },
                "evidence": [current_value],
            }
        ]
        base_scene = _parse_scene(
            json.dumps(payload, ensure_ascii=False),
            fingerprint="frame-committed-cue",
        )

        def parsed(*, cues: list[str], placeholder: str = ""):
            audit = input_audit_payload(
                application_inputs=[
                    audited_application_input(
                        structure_id="app-input-1",
                        bounds=[140, 530, 710, 590],
                        text="",
                        placeholder=placeholder,
                    )
                ],
                ime_preedit_regions=[],
                keyboard={
                    "visible": True,
                    "bounds": [0, 660, 1000, 1000],
                    "layout": "qwerty",
                    "input_mode": "direct_latin",
                    "case_mode": "lower",
                    "qwerty_anchors": {
                        "q": [122, 710], "p": [880, 710],
                        "a": [164, 782], "l": [838, 782],
                        "z": [248, 853], "m": [754, 853],
                        "backspace": [880, 853],
                    },
                    "mode_switch": None,
                    "backspace_key": None,
                    "case_switch": None,
                    "enter_key": None,
                    "literal_keys": [],
                    "layout_switches": [],
                },
            )
            audit["application_inputs"][0]["visible_editable_cues"] = cues
            return _apply_input_structure_audit(
                base_scene,
                json.dumps(audit, ensure_ascii=False),
                fingerprint="frame-committed-cue",
                goal_context={
                    "objective": "只发送一次输入框内现有正文",
                    "entities": {"input_text": current_value},
                },
                ledger_input_value=current_value,
            )

        rejected_shadow = parsed(cues=[current_value])
        field = rejected_shadow.get_element("local_audited_input_1")
        self.assertEqual("", field.states["value"])
        self.assertNotIn("ime_preedit_text", field.states)

        for cues, placeholder in (
            (["different"], ""),
            ([current_value, "different"], ""),
            ([current_value], current_value),
        ):
            with self.subTest(cues=cues, placeholder=placeholder):
                rejected = parsed(cues=cues, placeholder=placeholder)
                self.assertEqual(
                    "",
                    rejected.get_element("local_audited_input_1").states["value"],
                )

    def test_empty_optional_backspace_label_does_not_discard_valid_input(self) -> None:
        base_scene = _parse_scene(
            json.dumps(scene_payload(), ensure_ascii=False),
            fingerprint="frame-empty-backspace-label",
        )
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="message-field",
                    bounds=[153, 540, 690, 590],
                    text="",
                )
            ],
            keyboard={
                "visible": True,
                "bounds": [0, 600, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "case_mode": "lower",
                "qwerty_anchors": {
                    "q": [100, 730], "p": [890, 730],
                    "a": [140, 810], "l": [850, 810],
                    "z": [260, 890], "m": [750, 890],
                    "backspace": [890, 890],
                },
                "mode_switch": None,
                "backspace_key": {
                    "label": "",
                    "bounds": [830, 860, 960, 920],
                    "confidence": 1.0,
                    "fully_visible": True,
                },
                "case_switch": None,
                "literal_keys": [],
                "layout_switches": [],
            },
        )
        scene = _apply_input_structure_audit(
            base_scene,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="frame-empty-backspace-label",
            goal_context={
                "objective": "消息输入框最终只显示 stage，不要发送",
                "entities": {"input_text": "stage"},
            },
        )

        field = scene.get_element("local_audited_input_1")
        self.assertIsNotNone(field)
        self.assertEqual("", field.states["value"])
        self.assertEqual("direct_latin", field.states["keyboard_input_mode"])

        invalid = json.loads(json.dumps(audit, ensure_ascii=False))
        invalid["keyboard"]["backspace_key"]["label"] = "机器人"
        scene_without_backspace = _apply_input_structure_audit(
            base_scene,
            json.dumps(invalid, ensure_ascii=False),
            fingerprint="frame-invalid-backspace-label",
            goal_context={
                "objective": "消息输入框最终只显示 stage，不要发送",
                "entities": {"input_text": "stage"},
            },
        )
        self.assertEqual(
            "qwerty",
            scene_without_backspace.get_element(
                "local_audited_input_1"
            ).states["keyboard_geometry"]["type"],
        )

    def test_input_audit_mints_layout_and_case_switches_only_for_next_step(self) -> None:
        base_scene = _parse_scene(
            json.dumps(scene_payload(), ensure_ascii=False),
            fingerprint="frame-switches",
        )
        common_input = [
            audited_application_input(
                structure_id="message-field", bounds=[80, 120, 920, 210], text=""
            )
        ]
        upper_audit = input_audit_payload(
            application_inputs=common_input,
            keyboard={
                "visible": True, "bounds": [0, 480, 1000, 1000],
                "layout": "qwerty", "input_mode": "direct_latin",
                "case_mode": "lower",
                "qwerty_anchors": {
                    "q": [115, 610], "p": [875, 610],
                    "a": [157, 700], "l": [832, 700],
                    "z": [241, 790], "m": [747, 790],
                    "backspace": [875, 790],
                },
                "mode_switch": None,
                "case_switch": {
                    "label": "⇧", "bounds": [40, 760, 130, 850],
                    "confidence": 0.98, "current_mode": "lower",
                    "target_mode": "upper",
                },
                "literal_keys": [],
                "layout_switches": [
                    {
                        "label": "123", "bounds": [80, 880, 200, 980],
                        "confidence": 0.98, "current_layout": "qwerty",
                        "target_layout": "numeric",
                    }
                ],
            },
        )
        upper_scene = _apply_input_structure_audit(
            base_scene, json.dumps(upper_audit, ensure_ascii=False),
            fingerprint="frame-switches",
            goal_context={"objective": "草稿内容为Meeting", "entities": {"input_text": "Meeting"}},
        )
        self.assertEqual(
            "local_audited_keyboard_case_switch_1",
            upper_scene.unique_trusted_goal_element().element_id,
        )

        numeric_scene = _apply_input_structure_audit(
            base_scene, json.dumps(upper_audit, ensure_ascii=False),
            fingerprint="frame-switches",
            goal_context={"objective": "草稿内容为8", "entities": {"input_text": "8"}},
        )
        self.assertEqual(
            "local_audited_keyboard_layout_switch_1",
            numeric_scene.unique_trusted_goal_element().element_id,
        )

    def test_literal_key_and_layout_switch_validation_fail_closed(self) -> None:
        keyboard_bounds = (0.0, 480.0, 1000.0, 1000.0)
        self.assertEqual(
            [],
            _validated_keyboard_literal_keys(
                [{
                    "value": "8", "label": "9", "key_kind": "character",
                    "bounds": [200, 600, 280, 690], "confidence": 0.99,
                    "fully_visible": True,
                }],
                keyboard_bounds=keyboard_bounds,
            ),
        )
        self.assertEqual(
            [],
            _validated_keyboard_literal_keys(
                [{
                    "value": "8", "label": "8", "key_kind": "character",
                    "bounds": [200, 600, 280, 690], "confidence": 0.70,
                    "fully_visible": True,
                }],
                keyboard_bounds=keyboard_bounds,
            ),
        )
        self.assertEqual(
            [],
            _validated_keyboard_layout_switches(
                [{
                    "label": "emoji", "bounds": [80, 880, 200, 980],
                    "confidence": 0.99, "current_layout": "qwerty",
                    "target_layout": "numeric",
                }],
                keyboard_bounds=keyboard_bounds,
                current_layout="qwerty",
            ),
        )
        with self.assertRaisesRegex(UISceneError, "字段不符合协议"):
            _validated_keyboard_layout_switches(
                [
                    {
                        "label": "ABC",
                        "bounds": [80, 880, 200, 980],
                        "confidence": 0.99,
                        "current_layout": "symbol",
                        "target_layout": "qwerty",
                        "action": "tap",
                    }
                ],
                keyboard_bounds=keyboard_bounds,
                current_layout="symbol",
            )
        with self.assertRaisesRegex(UISceneError, "字段不符合协议"):
            _validated_keyboard_backspace_key(
                {
                    "label": "⌫",
                    "bounds": [800, 820, 940, 900],
                    "confidence": 1.0,
                    "fully_visible": True,
                    "action": "tap",
                },
                keyboard_bounds=keyboard_bounds,
            )
        self.assertEqual(
            [
                {
                    "label": "ABC",
                    "bounds": [80, 880, 200, 980],
                    "confidence": 0.99,
                    "current_layout": "symbol",
                    "target_layout": "qwerty",
                }
            ],
            _validated_keyboard_layout_switches(
                [
                    {
                        "label": "ABC",
                        "bounds": [80, 880, 200, 980],
                        "confidence": 0.99,
                        "current_layout": "symbol_grid",
                        "target_layout": "qwerty",
                    }
                ],
                keyboard_bounds=keyboard_bounds,
                current_layout="symbol",
            ),
        )
        self.assertEqual(
            [],
            _validated_keyboard_layout_switches(
                [
                    {
                        "label": "ABC",
                        "bounds": [80, 880, 200, 980],
                        "confidence": 0.99,
                        "current_layout": "symbols_custom",
                        "target_layout": "qwerty",
                    }
                ],
                keyboard_bounds=keyboard_bounds,
                current_layout="symbol",
            ),
        )

    def test_valid_literal_key_survives_invalid_optional_keyboard_claims(
        self,
    ) -> None:
        base_scene = _parse_scene(
            json.dumps(scene_payload(), ensure_ascii=False),
            fingerprint="frame-symbol-optional-claims",
        )
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="message-field",
                    bounds=[150, 530, 680, 590],
                    text="复杂输入验收2026",
                )
            ],
            keyboard={
                "visible": True,
                "bounds": [0, 600, 1000, 1000],
                "layout": "symbol_grid",
                "input_mode": "chinese_pinyin",
                "case_mode": "unknown",
                "qwerty_anchors": None,
                "mode_switch": None,
                "backspace_key": {
                    "label": "✘",
                    "bounds": [220, 820, 360, 880],
                    "confidence": 1.0,
                    "fully_visible": True,
                },
                "case_switch": None,
                "literal_keys": [
                    {
                        "value": "：",
                        "label": "：",
                        "key_kind": "character",
                        "bounds": [360, 660, 500, 720],
                        "confidence": 1.0,
                        "fully_visible": True,
                    }
                ],
                "layout_switches": [
                    {
                        "label": "常用",
                        "bounds": [80, 660, 220, 720],
                        "confidence": 1.0,
                        "current_layout": "symbol_grid",
                        "target_layout": "frequent",
                    },
                    {
                        "label": "返回",
                        "bounds": [80, 880, 220, 940],
                        "confidence": 1.0,
                        "current_layout": "symbol_grid",
                        "target_layout": "previous",
                    },
                ],
            },
        )

        scene = _apply_input_structure_audit(
            base_scene,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="frame-symbol-optional-claims",
            goal_context={
                "objective": "输入框最终逐字显示复杂输入验收2026：且不发送",
                "entities": {"input_text": "复杂输入验收2026："},
            },
        )

        target = scene.unique_trusted_goal_element()
        self.assertIsNotNone(target)
        self.assertEqual("local_audited_literal_key_1", target.element_id)
        self.assertEqual("：", target.states["key_value"])
        field = scene.get_element("local_audited_input_1")
        self.assertNotIn("keyboard_geometry", field.states)

    def test_hidden_keyboard_only_attestation_rejects_structured_keyboard_conflict(self) -> None:
        compact = scene_payload()
        compact["summary"] = "输入框中显示codex。"
        compact["overlays"] = ["软键盘"]
        compact["elements"] = [
            {
                "element_id": "model-input",
                "role": "input",
                "meaning": "application_text_input",
                "label": "codex",
                "bounds": [120, 820, 880, 900],
                "confidence": 0.98,
                "states": {
                    "goal_relevant": True,
                    "fully_visible": True,
                    "value": "codex",
                },
                "evidence": ["输入框内逐字显示codex"],
            }
        ]
        audit = input_audit_payload(
            application_inputs=[{"invalid": "geometry is ignored"}],
            keyboard={
                "visible": False,
                "bounds": None,
                "layout": "unknown",
                "input_mode": "unknown",
                "qwerty_anchors": None,
                "mode_switch": None,
            },
        )
        context = {
            "objective": "输入后收起软键盘",
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "hide_keyboard",
                    "objective": "软键盘已收起",
                    "constraints": [],
                    "completion_conditions": ["软键盘不可见"],
                    "execution_class": "navigate",
                    "goal_entities": {"input_text": "codex"},
                }
            },
        }

        with self.assertRaisesRegex(VisionAgentError, "输入结构只读审计"):
            GenericSceneObserver(SequenceProvider([compact, audit])).observe(
                frames=stable_frames(),
                goal_context=context,
            )

    def test_active_input_focus_ignores_completed_reload_wording(self) -> None:
        compact = scene_payload()
        compact["elements"] = [
            {
                "element_id": "model-input",
                "role": "input",
                "meaning": "application_text_input",
                "label": "",
                "bounds": [110, 40, 850, 110],
                "confidence": 0.96,
                "states": {
                    "goal_relevant": True,
                    "fully_visible": False,
                    "value": "",
                },
                "evidence": ["完整输入边框"],
            }
        ]
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(text="", placeholder="请输入")
            ],
        )
        provider = SequenceProvider([compact, audit])
        observer = GenericSceneObserver(provider)
        context = {
            "objective": "先重新加载页面，再确认输入框为空",
            "entities": {
                "original_goal_visual_context": "刷新后让输入框恢复为空",
                "active_subgoal_visual_context": {
                    "subgoal_id": "verify_input",
                    "objective": "确认唯一输入框为空",
                    "constraints": [],
                    "completion_conditions": ["输入框可见且文字为空"],
                    "execution_class": "observe",
                    "goal_entities": {},
                },
            },
        }

        scene = observer.observe(frames=stable_frames(), goal_context=context)

        self.assertEqual(2, provider.calls)
        self.assertFalse(observer.last_diagnostics["icon_cluster_audit_used"])
        self.assertTrue(observer.last_diagnostics["input_structure_audit_used"])
        self.assertEqual(
            "local_audited_input_1",
            scene.unique_trusted_goal_element().element_id,
        )

    def test_exact_input_payload_candidate_node_runs_strict_input_audit(self) -> None:
        compact = scene_payload()
        compact["screen_id"] = "chat_interface"
        compact["summary"] = "当前页面显示拼音候选栏和已聚焦的应用输入框。"
        compact["overlays"] = ["keyboard"]
        compact["elements"] = [
            {
                "element_id": "model-candidate",
                "role": "button",
                "meaning": "select_candidate",
                "label": "你好",
                "bounds": [80, 392, 220, 458],
                "confidence": 0.99,
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": ["拼音nihao对应的首个候选词"],
            },
            {
                "element_id": "model-input",
                "role": "input",
                "meaning": "message_input_box",
                "label": "",
                "bounds": [80, 120, 920, 210],
                "confidence": 0.98,
                "states": {
                    "goal_relevant": False,
                    "fully_visible": True,
                    "value": "",
                },
                "evidence": ["应用输入框当前为空"],
            },
        ]
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="message-field",
                    bounds=[80, 120, 920, 210],
                    text="",
                )
            ],
            ime_preedit_regions=[
                {
                    "region_id": "candidate-strip",
                    "bounds": [40, 380, 960, 470],
                    "text": "nihao",
                    "confidence": 0.98,
                    "candidates": [
                        {
                            "text": "你好",
                            "bounds": [80, 392, 220, 458],
                            "confidence": 0.98,
                            "fully_visible": True,
                        },
                        {
                            "text": "拟好",
                            "bounds": [250, 392, 390, 458],
                            "confidence": 0.96,
                            "fully_visible": True,
                        },
                    ],
                }
            ],
            keyboard={
                "visible": True,
                "bounds": [0, 480, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "chinese_pinyin",
                "qwerty_anchors": {
                    "q": [115, 610], "p": [875, 610],
                    "a": [157, 700], "l": [832, 700],
                    "z": [241, 790], "m": [747, 790],
                    "backspace": [875, 790],
                },
                "mode_switch": None,
            },
        )
        provider = SequenceProvider([compact, audit])
        observer = GenericSceneObserver(provider)
        context = {
            "objective": "完成当前未提交的中文输入",
            "entities": {
                "input_text": "你好",
                "target_ui_label": "你好",
                "active_subgoal_visual_context": {
                    "subgoal_id": "select_candidate",
                    "objective": "选择唯一逐字候选‘你好’",
                    "constraints": ["不要发送或提交"],
                    "completion_conditions": ["候选‘你好’被选中"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "input_text": "你好",
                        "target_ui_label": "你好",
                        "active_input_transaction_text": "你好",
                    },
                },
            },
        }

        scene = observer.observe(frames=stable_frames(), goal_context=context)

        self.assertEqual(2, provider.calls)
        self.assertTrue(observer.last_diagnostics["input_structure_audit_used"])
        candidate = scene.get_element("local_audited_ime_candidate_1")
        self.assertEqual("local_audited_ime_candidate_1", candidate.element_id)
        self.assertEqual("ime_exact_candidate", candidate.meaning)
        self.assertEqual("你好", candidate.label)
        self.assertEqual("你好", candidate.states["expected_input_value"])
        self.assertFalse(
            scene.get_element("model-candidate").states["goal_relevant"]
        )

    def test_other_active_label_does_not_reactivate_input_audit(self) -> None:
        compact = scene_payload()
        compact["elements"] = [
            {
                "element_id": "send-control",
                "role": "button",
                "meaning": "send_message",
                "label": "发送",
                "bounds": [780, 820, 940, 900],
                "confidence": 0.99,
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": ["输入框右侧唯一发送按钮"],
            }
        ]
        provider = SequenceProvider([compact])
        observer = GenericSceneObserver(provider)
        context = {
            "objective": "发送已准备的正文",
            "entities": {
                "input_text": "你好",
                "target_ui_label": "发送",
                "active_subgoal_visual_context": {
                    "subgoal_id": "send_message",
                    "objective": "发送已准备的正文",
                    "constraints": [],
                    "completion_conditions": ["正文已发送"],
                    "execution_class": "effect",
                    "goal_entities": {
                        "input_text": "你好",
                        "target_ui_label": "发送",
                    },
                },
            },
        }

        scene = observer.observe(frames=stable_frames(), goal_context=context)

        self.assertEqual(1, provider.calls)
        self.assertFalse(observer.last_diagnostics["input_structure_audit_used"])
        self.assertEqual("send-control", scene.unique_trusted_goal_element().element_id)

    def test_rejected_input_audit_without_attested_reload_remains_fail_closed(self) -> None:
        compact = scene_payload()
        compact["elements"] = [
            {
                "element_id": "model-reload",
                "role": "icon",
                "meaning": "reload",
                "label": "",
                "bounds": [820, 20, 875, 75],
                "confidence": 0.96,
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": ["右上方完整圆形箭头"],
            }
        ]
        invalid_input_audit = input_audit_payload(
            application_inputs=[audited_application_input(text="agent")],
            keyboard={
                "visible": True,
                "bounds": [0, 360, 1000, 1210],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "mode_switch": None,
            },
        )
        provider = SequenceProvider(
            [
                compact,
                icon_cluster_audit_payload(
                    controls=[],
                    cluster_complete=False,
                    cluster_bounds=None,
                ),
                invalid_input_audit,
            ]
        )

        with self.assertRaisesRegex(VisionAgentError, "可见键盘必须提供有效 bounds"):
            GenericSceneObserver(provider).observe(
                frames=icon_cluster_frames(),
                goal_context={
                    "objective": "重新加载当前页面，使唯一输入框恢复为空",
                },
            )

        self.assertEqual(3, provider.calls)

    def test_input_goal_removes_compact_keyboard_claim_and_uses_strict_audit(
        self,
    ) -> None:
        payload = scene_payload()
        payload["elements"] = [
            {
                "element_id": "bad-switch",
                "role": "button",
                "meaning": "switch_keyboard_input_mode",
                "label": "英",
                "bounds": [730, 1130, 810, 1210],
                "confidence": 0.92,
                "states": {
                    "goal_relevant": True,
                    "keyboard_input_mode_switch": True,
                    "current_mode": "direct_latin",
                    "target_mode": "direct_latin",
                },
                "evidence": ["方向与键面文字冲突"],
            }
        ]

        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(text="", placeholder="消息")
            ]
        )
        provider = SequenceProvider([payload, audit])
        scene = GenericSceneObserver(provider).observe(
            frames=stable_frames(),
            goal_context={
                "objective": "让当前输入框显示 agent",
                "entities": {"input_text": "agent"},
            },
        )
        self.assertEqual(2, provider.calls)
        self.assertEqual(
            "local_audited_input_1",
            scene.unique_trusted_goal_element().element_id,
        )
        self.assertFalse(
            any(
                item.meaning == "switch_keyboard_input_mode"
                for item in scene.elements
            )
        )

    def test_input_goal_does_not_hide_action_bearing_compact_keyboard_claim(
        self,
    ) -> None:
        payload = scene_payload()
        payload["elements"] = [
            {
                "element_id": "bad-switch",
                "role": "button",
                "meaning": "switch_keyboard_input_mode",
                "label": "英",
                "bounds": [730, 1130, 810, 1210],
                "confidence": 0.92,
                "states": {
                    "goal_relevant": True,
                    "keyboard_input_mode_switch": True,
                    "current_mode": "direct_latin",
                    "target_mode": "chinese_pinyin",
                    "action": "tap",
                },
                "evidence": ["协议外动作字段"],
            }
        ]

        with self.assertRaises(VisionAgentError):
            GenericSceneObserver(FakeProvider(payload)).observe(
                frames=stable_frames(),
                goal_context={
                    "objective": "让当前输入框显示 agent",
                    "entities": {"input_text": "agent"},
                },
            )

    def test_non_input_goal_cannot_hide_action_field_on_keyboard_switch(self) -> None:
        payload = scene_payload()
        payload["elements"] = [
            {
                "element_id": "bad-switch",
                "role": "button",
                "meaning": "switch_keyboard_input_mode",
                "label": "英",
                "bounds": [690, 890, 790, 930],
                "confidence": 0.92,
                "states": {
                    "goal_relevant": False,
                    "keyboard_input_mode_switch": True,
                    "current_mode": "direct_latin",
                    "target_mode": "chinese_pinyin",
                    "action": "tap",
                },
                "evidence": ["协议外动作字段"],
            }
        ]

        with self.assertRaises(VisionAgentError):
            GenericSceneObserver(FakeProvider(payload)).observe(
                frames=stable_frames(),
                goal_context={"objective": "当前页面完成一次重新加载"},
            )

    def test_reload_goal_keeps_only_literal_reload_control_relevant(self) -> None:
        payload = scene_payload()
        payload["elements"] = [
            {
                "element_id": "field",
                "role": "input",
                "meaning": "verification_input",
                "label": "agent",
                "bounds": [150, 440, 850, 530],
                "confidence": 0.98,
                "states": {"goal_relevant": True, "value": "agent"},
                "evidence": ["当前页面输入框"],
            },
            {
                "element_id": "refresh",
                "role": "button",
                "meaning": "refresh",
                "label": "",
                "bounds": [850, 20, 920, 90],
                "confidence": 0.98,
                "states": {"goal_relevant": False},
                "evidence": ["顶部右侧圆形箭头"],
            },
        ]

        provider = SequenceProvider(
            [
                payload,
                icon_cluster_audit_payload(),
                localized_icon_cluster_audit_payload(),
            ]
        )
        scene = GenericSceneObserver(provider).observe(
            frames=icon_cluster_frames(),
            goal_context={"objective": "当前页面完成一次重新加载"},
        )

        self.assertEqual(3, provider.calls)
        self.assertEqual(
            "local_audited_reload_control_1",
            scene.unique_trusted_goal_element().element_id,
        )
        self.assertFalse(scene.get_element("field").states["goal_relevant"])

    def test_reload_goal_uses_explicit_top_right_targeted_refinement(self) -> None:
        first = scene_payload()
        first["elements"] = [
            {
                "element_id": "field",
                "role": "input",
                "meaning": "verification_input",
                "label": "agent",
                "bounds": [150, 440, 850, 530],
                "confidence": 0.98,
                "states": {"goal_relevant": True, "value": "agent"},
                "evidence": ["当前页面输入框"],
            }
        ]
        refined = scene_payload()
        refined["elements"] = [
            {
                "element_id": "refresh",
                "role": "button",
                "meaning": "refresh",
                "label": "",
                "bounds": [850, 20, 920, 90],
                "confidence": 0.98,
                "states": {"goal_relevant": True},
                "evidence": ["顶部右侧圆形箭头"],
            }
        ]
        provider = SequenceProvider(
            [
                first,
                refined,
                icon_cluster_audit_payload(),
                localized_icon_cluster_audit_payload(),
            ]
        )
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=icon_cluster_frames(),
            goal_context={
                "objective": "圆形箭头对应的页面重新加载已完成",
                "entities": {"spatial_hint": "top_right"},
            },
        )

        self.assertEqual(4, provider.calls)
        self.assertTrue(observer.last_diagnostics["targeted_refinement_used"])
        self.assertEqual([440, 0, 1000, 420], observer.last_diagnostics["targeted_roi_bounds"])
        self.assertEqual(
            "local_audited_reload_control_1",
            scene.unique_trusted_goal_element().element_id,
        )
        targeted_prompt = json.dumps(provider.messages_seen[1], ensure_ascii=False)
        self.assertIn("一个element只能紧框一个", targeted_prompt)
        self.assertIn("相邻非目标图标", targeted_prompt)

    def test_low_confidence_clipped_goal_element_still_refines(self) -> None:
        first = scene_payload()
        first["elements"] = [
            {
                "element_id": "uncertain_refresh",
                "role": "icon",
                "meaning": "reload",
                "label": "圆形箭头图标",
                "bounds": [750, 0, 850, 50],
                "confidence": 0.65,
                "states": {"goal_relevant": True, "fully_visible": False},
                "evidence": ["局部图中疑似刷新控件"],
            }
        ]
        refined = scene_payload()
        refined["elements"] = [
            {
                "element_id": "refresh",
                "role": "button",
                "meaning": "reload",
                "label": "刷新",
                "bounds": [820, 10, 875, 55],
                "confidence": 0.96,
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": ["完整圆形箭头与相邻书签图标可区分"],
            }
        ]
        provider = SequenceProvider(
            [
                first,
                refined,
                icon_cluster_audit_payload(),
                localized_icon_cluster_audit_payload(),
            ]
        )
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=icon_cluster_frames(),
            goal_context={
                "objective": "圆形箭头对应的页面重新加载已完成",
                "entities": {"spatial_hint": "top_right"},
            },
        )

        self.assertEqual(4, provider.calls)
        self.assertTrue(observer.last_diagnostics["targeted_refinement_used"])
        self.assertEqual([440, 0, 1000, 420], observer.last_diagnostics["targeted_roi_bounds"])
        self.assertEqual(
            "local_audited_reload_control_1",
            scene.unique_trusted_goal_element().element_id,
        )

    def test_icon_cluster_audit_is_only_source_of_reload_attestation(self) -> None:
        compact = scene_payload()
        compact["elements"] = [
            {
                "element_id": "model-refresh",
                "role": "icon",
                "meaning": "reload",
                "label": "",
                "bounds": [820, 20, 875, 75],
                "confidence": 0.98,
                "states": {
                    "goal_relevant": True,
                    "fully_visible": True,
                    "reload_visual_audit": True,
                    "independent_geometry_verified": True,
                    "geometry_audit_source": "icon_cluster_localization",
                },
                "evidence": ["模型直接声称本地凭据"],
            }
        ]
        provider = SequenceProvider(
            [
                compact,
                icon_cluster_audit_payload(),
                localized_icon_cluster_audit_payload(),
            ]
        )
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=icon_cluster_frames(),
            goal_context={"objective": "刷新当前页面"},
        )

        candidate = scene.unique_trusted_goal_element()
        self.assertEqual(3, provider.calls)
        self.assertEqual("local_audited_reload_control_1", candidate.element_id)
        self.assertEqual(
            {
                "goal_relevant": True,
                "fully_visible": True,
                "reload_visual_audit": True,
                "independent_geometry_verified": True,
                "geometry_audit_source": "icon_cluster_localization",
            },
            candidate.states,
        )
        self.assertTrue(observer.last_diagnostics["icon_cluster_audit_used"])
        self.assertTrue(
            observer.last_diagnostics["icon_cluster_audit_reload_attested"]
        )

    def test_icon_cluster_localization_replaces_wrong_full_frame_bounds(self) -> None:
        compact = scene_payload()
        compact["elements"] = [
            {
                "element_id": "model-refresh",
                "role": "icon",
                "meaning": "reload",
                "label": "",
                "bounds": [880, 15, 950, 50],
                "confidence": 0.98,
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": ["粗观察的右上圆形箭头"],
            }
        ]
        rough = icon_cluster_audit_payload(
            controls=[
                {
                    "control_id": "rough-reload",
                    "semantic_class": "reload",
                    "bounds": [880, 15, 950, 50],
                    "confidence": 0.96,
                    "fully_visible": True,
                    "single_glyph": True,
                    "shape_cues": ["curved_arc", "arrowhead"],
                },
                {
                    "control_id": "rough-bookmark",
                    "semantic_class": "bookmark",
                    "bounds": [760, 15, 820, 50],
                    "confidence": 0.96,
                    "fully_visible": True,
                    "single_glyph": True,
                    "shape_cues": ["bookmark_outline"],
                },
            ],
            cluster_bounds=[740, 0, 970, 100],
        )
        localized = icon_cluster_audit_payload(
            controls=[
                {
                    "control_id": "local-reload",
                    "semantic_class": "reload",
                    "bounds": [750, 120, 950, 400],
                    "confidence": 0.97,
                    "fully_visible": True,
                    "single_glyph": True,
                    "shape_cues": ["curved_arc", "arrowhead"],
                },
                {
                    "control_id": "local-bookmark",
                    "semantic_class": "bookmark",
                    "bounds": [100, 120, 300, 400],
                    "confidence": 0.97,
                    "fully_visible": True,
                    "single_glyph": True,
                    "shape_cues": ["bookmark_outline"],
                },
            ],
            cluster_bounds=[0, 0, 1000, 800],
        )
        provider = SequenceProvider([compact, rough, localized])
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=icon_cluster_frames(),
            goal_context={"objective": "刷新当前页面"},
        )

        candidate = scene.unique_trusted_goal_element()
        center_x = (candidate.bounds[0] + candidate.bounds[2]) / 2.0
        self.assertGreater(center_x, 0.83)
        self.assertLess(center_x, 0.88)
        self.assertTrue(
            observer.last_diagnostics["icon_cluster_local_geometry_verified"]
        )
        self.assertTrue(observer.last_diagnostics["icon_cluster_localization_used"])
        self.assertEqual(
            [682, 0, 1000, 125],
            observer.last_diagnostics["icon_cluster_localization_roi_bounds"],
        )
        second_prompt = json.dumps(provider.messages_seen[2], ensure_ascii=False)
        self.assertIn("coordinate system is local to this crop", second_prompt)

    def test_icon_cluster_localization_failure_never_reuses_rough_bounds(self) -> None:
        compact = scene_payload()
        compact["elements"] = []
        rough = icon_cluster_audit_payload()
        localized_failure = icon_cluster_audit_payload(
            controls=[],
            cluster_complete=False,
        )
        provider = SequenceProvider([compact, compact, rough, localized_failure])
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={"objective": "点击顶部右侧刷新页面"},
        )

        self.assertIsNone(scene.unique_trusted_goal_element())
        self.assertTrue(observer.last_diagnostics["icon_cluster_localization_used"])
        self.assertFalse(
            observer.last_diagnostics["icon_cluster_audit_reload_attested"]
        )

    def test_local_ordinal_binding_snaps_every_control_not_only_reload(self) -> None:
        payload = icon_cluster_audit_payload(
            controls=[
                {
                    "control_id": "shifted-bookmark",
                    "semantic_class": "bookmark",
                    "bounds": [700, 120, 780, 400],
                    "confidence": 0.95,
                    "fully_visible": True,
                    "single_glyph": True,
                    "shape_cues": ["bookmark_outline"],
                },
                {
                    "control_id": "shifted-reload",
                    "semantic_class": "reload",
                    "bounds": [900, 120, 990, 400],
                    "confidence": 0.95,
                    "fully_visible": True,
                    "single_glyph": True,
                    "shape_cues": ["curved_arc", "arrowhead"],
                },
            ],
            cluster_bounds=[645, 0, 1000, 500],
        )

        snapped, reload_bounds = _snap_reload_audit_to_local_glyph(
            icon_cluster_frames()[0],
            payload,
            search_bounds=(645, 0, 975, 125),
        )

        self.assertIsNotNone(snapped)
        bookmark_bounds = snapped["controls"][0]["bounds"]
        self.assertLess(bookmark_bounds[2], reload_bounds[0])
        self.assertGreater((reload_bounds[0] + reload_bounds[2]) / 2.0, 830)
        self.assertLess((reload_bounds[0] + reload_bounds[2]) / 2.0, 880)

    def test_overlapping_reload_and_bookmark_cluster_fails_closed(self) -> None:
        compact = scene_payload()
        compact["elements"] = []
        audit = icon_cluster_audit_payload(
            controls=[
                {
                    "control_id": "reload",
                    "semantic_class": "reload",
                    "bounds": [760, 20, 850, 80],
                    "confidence": 0.97,
                    "fully_visible": True,
                    "single_glyph": True,
                    "shape_cues": ["curved_arc", "arrowhead"],
                },
                {
                    "control_id": "bookmark",
                    "semantic_class": "bookmark",
                    "bounds": [830, 20, 890, 80],
                    "confidence": 0.98,
                    "fully_visible": True,
                    "single_glyph": True,
                    "shape_cues": ["bookmark_outline"],
                },
            ]
        )
        provider = SequenceProvider([compact, compact, audit])
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={"objective": "点击顶部右侧刷新页面"},
        )

        self.assertEqual(3, provider.calls)
        self.assertIsNone(scene.unique_trusted_goal_element())
        self.assertFalse(
            observer.last_diagnostics["icon_cluster_audit_reload_attested"]
        )

    def test_multiple_reload_candidates_fail_closed(self) -> None:
        compact = scene_payload()
        compact["elements"] = []
        first_reload = icon_cluster_audit_payload()["controls"][0]
        second_reload = dict(first_reload)
        second_reload.update(
            {"control_id": "reload-control-2", "bounds": [900, 20, 950, 75]}
        )
        audit = icon_cluster_audit_payload(
            controls=[first_reload, second_reload],
            cluster_bounds=[780, 0, 970, 100],
        )
        provider = SequenceProvider([compact, compact, audit])

        scene = GenericSceneObserver(provider).observe(
            frames=stable_frames(),
            goal_context={"objective": "点击刷新页面"},
        )

        self.assertIsNone(scene.unique_trusted_goal_element())

    def test_model_authored_reload_attestation_is_stripped_when_audit_fails(self) -> None:
        compact = scene_payload()
        compact["elements"] = [
            {
                "element_id": "model-refresh",
                "role": "icon",
                "meaning": "reload",
                "label": "",
                "bounds": [820, 20, 875, 75],
                "confidence": 0.99,
                "states": {
                    "goal_relevant": True,
                    "fully_visible": True,
                    "reload_visual_audit": True,
                    "independent_geometry_verified": True,
                    "geometry_audit_source": "icon_cluster_localization",
                },
                "evidence": ["模型自称已审计"],
            }
        ]
        failed_audit = icon_cluster_audit_payload(
            controls=[],
            cluster_complete=False,
        )
        provider = SequenceProvider([compact, failed_audit])

        scene = GenericSceneObserver(provider).observe(
            frames=stable_frames(),
            goal_context={"objective": "刷新当前页面"},
        )

        self.assertIsNone(scene.unique_trusted_goal_element())
        self.assertFalse(
            any(
                element.states.get("reload_visual_audit") is True
                or element.states.get("independent_geometry_verified") is True
                or "geometry_audit_source" in element.states
                for element in scene.elements
            )
        )

    def test_bookmark_shape_cannot_be_attested_as_reload(self) -> None:
        compact = scene_payload()
        compact["elements"] = []
        audit = icon_cluster_audit_payload(
            controls=[
                {
                    "control_id": "wrong-reload",
                    "semantic_class": "reload",
                    "bounds": [800, 20, 860, 80],
                    "confidence": 0.99,
                    "fully_visible": True,
                    "single_glyph": True,
                    "shape_cues": ["bookmark_outline"],
                }
            ],
            cluster_bounds=[760, 0, 900, 100],
        )
        provider = SequenceProvider([compact, compact, audit])

        scene = GenericSceneObserver(provider).observe(
            frames=stable_frames(),
            goal_context={"objective": "点击顶部右侧圆形箭头刷新页面"},
        )

        self.assertIsNone(scene.unique_trusted_goal_element())

    def test_expand_shape_conflict_cannot_be_attested_as_reload(self) -> None:
        compact = scene_payload()
        compact["elements"] = []
        audit = icon_cluster_audit_payload(
            controls=[
                {
                    "control_id": "wrong-reload",
                    "semantic_class": "reload",
                    "bounds": [800, 20, 860, 80],
                    "confidence": 0.99,
                    "fully_visible": True,
                    "single_glyph": True,
                    "shape_cues": [
                        "curved_arc",
                        "arrowhead",
                        "four_corner_brackets",
                    ],
                }
            ],
            cluster_bounds=[760, 0, 900, 100],
        )
        provider = SequenceProvider([compact, compact, audit])

        scene = GenericSceneObserver(provider).observe(
            frames=stable_frames(),
            goal_context={"objective": "点击顶部右侧圆形箭头刷新页面"},
        )

        self.assertIsNone(scene.unique_trusted_goal_element())

    def test_icon_cluster_audit_rejects_duplicate_json_keys(self) -> None:
        compact = scene_payload()
        compact["elements"] = [
            {
                "element_id": "refresh",
                "role": "icon",
                "meaning": "reload",
                "label": "",
                "bounds": [820, 20, 875, 75],
                "confidence": 0.98,
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": ["圆形箭头"],
            }
        ]
        raw = json.dumps(icon_cluster_audit_payload(), ensure_ascii=False)
        raw = raw.replace(
            '"cluster_complete": true',
            '"cluster_complete": true, "cluster_complete": true',
            1,
        )
        provider = SequenceProvider([compact, raw])

        with self.assertRaisesRegex(VisionAgentError, "重复 JSON 字段"):
            GenericSceneObserver(provider).observe(
                frames=stable_frames(),
                goal_context={"objective": "刷新当前页面"},
            )

    def test_icon_cluster_audit_rejects_unknown_shape_cue(self) -> None:
        compact = scene_payload()
        compact["elements"] = []
        audit = icon_cluster_audit_payload()
        audit["controls"][0]["shape_cues"].append("magic_reload")
        provider = SequenceProvider([compact, compact, audit])

        with self.assertRaisesRegex(VisionAgentError, "shape_cues"):
            GenericSceneObserver(provider).observe(
                frames=stable_frames(),
                goal_context={"objective": "点击顶部右侧刷新页面"},
            )

    def test_icon_cluster_audit_repairs_only_redundant_control_envelope(self) -> None:
        payload = icon_cluster_audit_payload(
            cluster_bounds=[800, 10, 900, 60],
        )

        parsed = _strict_icon_cluster_audit_payload(
            json.dumps(payload, ensure_ascii=False)
        )

        self.assertEqual([740.0, 20.0, 875.0, 75.0], parsed["cluster_bounds"])

    def test_text_entry_does_not_discard_incomplete_switch_without_direct_latin(self) -> None:
        empty = scene_payload()
        empty["elements"] = []
        audit = input_audit_payload(
            application_inputs=[audited_application_input(text="", placeholder="")],
            keyboard={
                "visible": True,
                "bounds": [0, 360, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "unknown",
                "mode_switch": {
                    "label": "英",
                    "bounds": [650, 900, 760, 970],
                    "confidence": 0.97,
                    "current_mode": "direct_latin",
                },
            },
        )
        audit["application_inputs"][0]["visible_editable_cues"] = ["caret"]

        with self.assertRaisesRegex(VisionAgentError, "mode_switch.*字段"):
            GenericSceneObserver(
                SequenceProvider([empty, empty, audit])
            ).observe(
                frames=stable_frames(),
                goal_context={"objective": "让当前唯一空白输入框显示 agent，不提交"},
            )

    def test_ordinary_letter_key_cannot_become_keyboard_mode_switch(self) -> None:
        empty = scene_payload()
        empty["elements"] = []
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    bounds=[80, 35, 820, 115],
                    text="",
                    placeholder="搜索",
                )
            ],
            keyboard={
                "visible": True,
                "bounds": [0, 360, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "chinese_pinyin",
                "mode_switch": {
                    "label": "A",
                    "bounds": [650, 900, 760, 970],
                    "confidence": 0.97,
                    "current_mode": "chinese_pinyin",
                    "target_mode": "direct_latin",
                },
            },
        )
        scene = GenericSceneObserver(
            SequenceProvider([empty, audit])
        ).observe(
            frames=stable_frames(),
            goal_context={"objective": "切换到英文直输模式"},
        )

        self.assertIsNone(scene.unique_trusted_goal_element())
        self.assertFalse(
            any(item.meaning == "switch_keyboard_input_mode" for item in scene.elements)
        )

    def test_targeted_refinement_cannot_publish_input_without_dedicated_audit(
        self,
    ) -> None:
        first = scene_payload()
        first["elements"] = []
        refined = scene_payload()
        refined["elements"] = [
            {
                "element_id": "input1",
                "role": "input",
                "meaning": "search_query_input",
                "label": "已有查询文字",
                "bounds": [100, 100, 900, 300],
                "confidence": 0.97,
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": ["已有查询文字"],
            }
        ]
        provider = SequenceProvider([first, refined, input_audit_payload()])
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={
                "objective": "修改已有文字的输入框",
                "entities": {"spatial_hint": "top"},
            },
        )

        self.assertFalse(scene.elements)
        self.assertEqual(
            "typed输入状态账本未建立；compact输入摘要与输入转写不参与判断。",
            scene.summary,
        )
        self.assertEqual(observer.last_diagnostics["targeted_roi_bounds"], [0, 0, 1000, 420])
        compact_image = provider.messages_seen[0][1]["content"][1]["image_url"]["url"]
        targeted_overview = provider.messages_seen[1][1]["content"][1]["image_url"]["url"]
        self.assertEqual(compact_image, targeted_overview)
        self.assertEqual(2, len(provider.messages_seen[1][1]["content"]))
        targeted_text = provider.messages_seen[1][1]["content"][0]["text"]
        self.assertIn("本次仍提供完整手机画面", targeted_text)
        self.assertNotIn("第二张高清局部", targeted_text)

    def test_no_spatial_goal_keeps_full_frame_for_targeted_refinement(self) -> None:
        first = scene_payload()
        first["elements"] = []
        refined = scene_payload()
        provider = SequenceProvider([first, refined, input_audit_payload()])
        observer = GenericSceneObserver(provider)

        observer.observe(
            frames=stable_frames(),
            goal_context={"objective": "查找已有文字的输入框"},
        )

        compact_image = provider.messages_seen[0][1]["content"][1]["image_url"]["url"]
        targeted_image = provider.messages_seen[1][1]["content"][1]["image_url"]["url"]
        self.assertEqual(compact_image, targeted_image)
        self.assertEqual(2, len(provider.messages_seen[1][1]["content"]))
        self.assertIsNone(observer.last_diagnostics["targeted_roi_bounds"])

    def test_strict_field_text_submit_structure_normalizes_to_one_input(self) -> None:
        payload = scene_payload()
        payload["elements"] = [
            {
                "element_id": "bar",
                "role": "container",
                "meaning": "search_bar_container",
                "label": "搜索栏",
                "bounds": [100, 80, 900, 180],
                "confidence": 0.96,
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": ["横向边框"],
            },
            {
                "element_id": "query",
                "role": "text",
                "meaning": "current_query_text",
                "label": "已有文字",
                "bounds": [180, 105, 560, 155],
                "confidence": 0.98,
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": ["已有文字"],
            },
            {
                "element_id": "submit",
                "role": "button",
                "meaning": "search_submit_button",
                "label": "搜索",
                "bounds": [700, 80, 900, 180],
                "confidence": 0.97,
                "states": {"goal_relevant": False, "fully_visible": True},
                "evidence": ["独立按钮"],
            },
        ]
        observer = GenericSceneObserver(FakeProvider(payload))

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={"objective": "修改顶部搜索输入框中的文字"},
        )

        candidate = scene.unique_trusted_goal_element()
        self.assertIsNotNone(candidate)
        self.assertEqual("input", candidate.role)
        self.assertEqual("已有文字", candidate.label)
        self.assertEqual((0.1, 0.08, 0.7, 0.18), candidate.bounds)
        self.assertTrue(observer.last_diagnostics["prefilled_input_structure_inferred"])
        self.assertFalse(scene.get_element("submit").states["goal_relevant"])

    def test_input_structure_is_not_inferred_without_explicit_input_goal(self) -> None:
        payload = scene_payload()
        payload["elements"] = [
            {
                "element_id": "bar",
                "role": "container",
                "meaning": "search_bar_container",
                "label": "搜索栏",
                "bounds": [100, 80, 900, 180],
                "confidence": 0.96,
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": ["横向边框"],
            },
            {
                "element_id": "query",
                "role": "text",
                "meaning": "current_query_text",
                "label": "已有文字",
                "bounds": [180, 105, 560, 155],
                "confidence": 0.98,
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": ["已有文字"],
            },
            {
                "element_id": "submit",
                "role": "button",
                "meaning": "search_submit_button",
                "label": "搜索",
                "bounds": [700, 80, 900, 180],
                "confidence": 0.97,
                "states": {"goal_relevant": False, "fully_visible": True},
                "evidence": ["独立按钮"],
            },
        ]
        observer = GenericSceneObserver(FakeProvider(payload))

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={"objective": "查看顶部区域"},
        )

        self.assertFalse(any(item.role == "input" for item in scene.elements))
        self.assertFalse(observer.last_diagnostics["prefilled_input_structure_inferred"])

    def test_clipped_structure_is_never_normalized_to_input(self) -> None:
        payload = scene_payload()
        payload["elements"] = [
            {
                "element_id": "field",
                "role": "container",
                "meaning": "query_input_container",
                "label": "裁切查询区域",
                "bounds": [110, 1, 740, 70],
                "confidence": 0.99,
                "states": {"goal_relevant": True, "fully_visible": False},
                "evidence": ["上边缘被画面裁切"],
            },
            {
                "element_id": "query",
                "role": "text",
                "meaning": "current_query_text",
                "label": "已有文字",
                "bounds": [200, 5, 475, 50],
                "confidence": 0.99,
                "states": {"goal_relevant": True, "fully_visible": False},
                "evidence": ["文字贴近上边缘"],
            },
            {
                "element_id": "submit",
                "role": "button",
                "meaning": "search_action_button",
                "label": "搜索",
                "bounds": [700, 1, 830, 70],
                "confidence": 0.99,
                "states": {"goal_relevant": False, "fully_visible": False},
                "evidence": ["按钮上边缘被裁切"],
            },
        ]
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="clipped",
                    bounds=[110, 1, 830, 70],
                    fully_visible=False,
                    confidence=0.99,
                    right_button={
                        "label": "搜索",
                        "bounds": [700, 1, 830, 70],
                        "confidence": 0.99,
                    },
                )
            ]
        )
        observer = GenericSceneObserver(SequenceProvider([payload, audit]))

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={"objective": "修改顶部搜索输入框中的文字"},
        )

        self.assertFalse(any(item.role == "input" for item in scene.elements))
        self.assertFalse(observer.last_diagnostics["prefilled_input_structure_inferred"])

    def test_input_audit_selects_only_complete_unique_structure_in_full_frame_coordinates(self) -> None:
        empty = scene_payload()
        empty["elements"] = []
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="complete",
                    bounds=[108, 78, 836, 129],
                    text="已有查询文字",
                    right_button={
                        "label": "搜索",
                        "bounds": [704, 78, 836, 129],
                        "confidence": 0.97,
                    },
                ),
                audited_application_input(
                    structure_id="clipped",
                    bounds=[108, 1, 836, 45],
                    fully_visible=False,
                    text="已有查询文字",
                    confidence=0.99,
                    right_button={
                        "label": "搜索",
                        "bounds": [750, 1, 836, 45],
                        "confidence": 0.99,
                    },
                ),
            ]
        )
        provider = SequenceProvider([empty, empty, audit])
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={"objective": "修改顶部搜索输入框中的文字"},
        )

        candidate = scene.unique_trusted_goal_element()
        self.assertIsNotNone(candidate)
        self.assertEqual("local_audited_input_1", candidate.element_id)
        self.assertEqual((0.108, 0.078, 0.704, 0.129), candidate.bounds)
        button = scene.get_element("local_audited_adjacent_button_1")
        self.assertEqual((0.704, 0.078, 0.836, 0.129), button.bounds)
        self.assertFalse(button.states["goal_relevant"])
        self.assertTrue(observer.last_diagnostics["input_structure_audit_used"])
        self.assertEqual([2600, 2600, 2600], provider.max_tokens_seen)

    def test_input_audit_budget_accepts_complete_verbose_keyboard_structure(self) -> None:
        empty = scene_payload()
        empty["elements"] = []
        candidates = [
            {
                "text": f"candidate-{index}",
                "bounds": [40 + index * 110, 600, 130 + index * 110, 645],
                "confidence": 0.98,
                "fully_visible": True,
            }
            for index in range(8)
        ]
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    bounds=[150, 530, 700, 590],
                    text="codex",
                    right_button={
                        "label": "发送",
                        "bounds": [780, 530, 920, 590],
                        "confidence": 0.98,
                    },
                )
            ],
            ime_preedit_regions=[
                {
                    "region_id": "ime-preedit-1",
                    "bounds": [0, 590, 1000, 650],
                    "text": "",
                    "confidence": 0.98,
                    "candidates": candidates,
                }
            ],
            keyboard={
                "visible": True,
                "bounds": [0, 650, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "case_mode": "lower",
                "qwerty_anchors": {
                    "q": [80, 730],
                    "p": [920, 730],
                    "a": [150, 810],
                    "l": [850, 810],
                    "z": [250, 890],
                    "m": [750, 890],
                    "backspace": [880, 890],
                },
                "mode_switch": {
                    "label": "英",
                    "bounds": [750, 930, 820, 980],
                    "confidence": 0.98,
                    "current_mode": "direct_latin",
                    "target_mode": "chinese_pinyin",
                },
                "backspace_key": {
                    "label": "⌫",
                    "bounds": [850, 870, 950, 910],
                    "confidence": 0.98,
                    "fully_visible": True,
                },
                "case_switch": {
                    "label": "⇧",
                    "bounds": [40, 870, 130, 920],
                    "confidence": 0.98,
                    "current_mode": "lower",
                    "target_mode": "upper",
                },
                "literal_keys": [],
                "layout_switches": [
                    {
                        "label": "123",
                        "bounds": [120, 930, 220, 980],
                        "confidence": 0.98,
                        "current_layout": "qwerty",
                        "target_layout": "numeric",
                    }
                ],
            },
        )
        verbose_response = json.dumps(audit, ensure_ascii=False, indent=2)
        self.assertGreater(len(verbose_response), 2600)
        provider = SequenceProvider([empty, empty, verbose_response])

        scene = GenericSceneObserver(provider).observe(
            frames=stable_frames(),
            goal_context={"objective": "清空当前输入框中的旧文字"},
        )

        candidate = scene.unique_trusted_goal_element()
        self.assertIsNotNone(candidate)
        self.assertEqual("codex", candidate.states["value"])
        self.assertEqual([2600, 2600, 2600], provider.max_tokens_seen)

    def test_incomplete_disjoint_right_button_is_discarded_without_input_widening(self) -> None:
        empty = scene_payload()
        empty["elements"] = []
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    bounds=[190, 530, 720, 590],
                    text="codex",
                    right_button={
                        "label": "发送",
                        "bounds": [790, 530, 930, 590],
                    },
                )
            ]
        )
        scene = GenericSceneObserver(
            SequenceProvider([empty, empty, audit])
        ).observe(
            frames=stable_frames(),
            goal_context={
                "objective": "确认底部输入框中的 codex 草稿仍可见",
                "entities": {"input_text": "codex"},
            },
        )

        candidate = scene.unique_trusted_goal_element()
        self.assertEqual("local_audited_input_1", candidate.element_id)
        self.assertEqual((0.19, 0.53, 0.72, 0.59), candidate.bounds)
        self.assertEqual("codex", candidate.states["value"])
        self.assertFalse(
            any(item.element_id == "local_audited_adjacent_button_1" for item in scene.elements)
        )

    def test_disjoint_right_button_with_visibility_fact_is_discarded(self) -> None:
        empty = scene_payload()
        empty["elements"] = []
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    bounds=[290, 570, 680, 610],
                    text="aaazjie",
                    right_button={
                        "label": "发送",
                        "bounds": [690, 570, 760, 610],
                        "confidence": 1.0,
                        "fully_visible": True,
                    },
                )
            ]
        )

        scene = GenericSceneObserver(
            SequenceProvider([empty, empty, audit])
        ).observe(
            frames=stable_frames(),
            goal_context={
                "objective": "确认当前输入内容",
                "entities": {"input_text": "aaazjie"},
            },
        )

        candidate = scene.unique_trusted_goal_element()
        self.assertEqual("local_audited_input_1", candidate.element_id)
        self.assertEqual("aaazjie", candidate.states["value"])
        self.assertFalse(
            any(item.element_id == "local_audited_adjacent_button_1" for item in scene.elements)
        )

    def test_incomplete_right_button_inside_combined_bounds_remains_invalid(self) -> None:
        empty = scene_payload()
        empty["elements"] = []
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    bounds=[190, 530, 930, 590],
                    text="codex",
                    right_button={
                        "label": "发送",
                        "bounds": [790, 530, 930, 590],
                    },
                )
            ]
        )

        with self.assertRaisesRegex(VisionAgentError, "right_button"):
            GenericSceneObserver(SequenceProvider([empty, empty, audit])).observe(
                frames=stable_frames(),
                goal_context={"objective": "确认输入框中的 codex"},
            )

    def test_top_obstruction_prevents_audit_crop_from_promoting_hidden_input(self) -> None:
        empty = scene_payload()
        empty["elements"] = []
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    bounds=[80, 12, 760, 86],
                    text="",
                    placeholder="搜索",
                )
            ],
            keyboard={
                "visible": True,
                "bounds": [40, 560, 960, 980],
                "layout": "qwerty",
                "input_mode": "chinese_pinyin",
                "mode_switch": {
                    "label": "中",
                    "bounds": [700, 890, 770, 940],
                    "confidence": 0.97,
                    "current_mode": "chinese_pinyin",
                    "target_mode": "direct_latin",
                },
            },
        )
        observer = GenericSceneObserver(SequenceProvider([empty, empty, audit]))

        scene = observer.observe(
            frames=frames_with_top_obstruction(),
            goal_context={"objective": "切换顶部输入框的输入模式"},
        )

        inputs = [item for item in scene.elements if item.role == "input"]
        self.assertEqual(1, len(inputs))
        self.assertFalse(inputs[0].states["fully_visible"])
        self.assertFalse(inputs[0].states["goal_relevant"])
        self.assertNotIn("focused", inputs[0].states)
        self.assertIsNone(scene.unique_trusted_goal_element())
        self.assertEqual(
            "top_edge_opaque_band",
            observer.last_diagnostics["visual_obstructions"][0]["kind"],
        )
        self.assertTrue(any("顶部不透明视觉遮挡" in item for item in scene.overlays))

    def test_input_audit_refuses_multiple_complete_structures(self) -> None:
        empty = scene_payload()
        empty["elements"] = []
        structure = audited_application_input(
            structure_id="one",
            bounds=[100, 100, 900, 180],
            right_button={
                "label": "搜索",
                "bounds": [720, 100, 900, 180],
                "confidence": 0.98,
            },
        )
        second = json.loads(json.dumps(structure, ensure_ascii=False))
        second["structure_id"] = "two"
        second["bounds"] = [100, 240, 900, 320]
        second["right_button"]["bounds"] = [720, 240, 900, 320]
        observer = GenericSceneObserver(
            SequenceProvider(
                [
                    empty,
                    empty,
                    input_audit_payload(application_inputs=[structure, second]),
                ]
            )
        )

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={"objective": "修改顶部搜索输入框中的文字"},
        )

        self.assertIsNone(scene.unique_trusted_goal_element())
        self.assertFalse(any(item.role == "input" for item in scene.elements))

    def test_multifield_audit_selects_visible_field_and_mints_only_newline_enter(self) -> None:
        base = _parse_scene(
            json.dumps(scene_payload(), ensure_ascii=False),
            fingerprint="f" * 64,
        )
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "fill_body",
                    "objective": "正文内容逐字等于目标文本",
                    "constraints": [],
                    "completion_conditions": [],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "第一行\n第二行",
                        "active_input_field_id": "body",
                        "active_input_field_label": "正文",
                        "active_input_multiline": True,
                    },
                }
            }
        }
        common_keyboard = {
            "visible": True,
            "bounds": [40, 560, 960, 980],
            "layout": "qwerty",
            "input_mode": "chinese_pinyin",
            "mode_switch": None,
            "enter_key": {
                "label": "↵",
                "bounds": [800, 850, 930, 940],
                "confidence": 0.98,
                "fully_visible": True,
                "key_action": "newline",
            },
        }
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="subject",
                    bounds=[100, 160, 900, 235],
                    text="主题值",
                    field_labels=["主题"],
                ),
                audited_application_input(
                    structure_id="body",
                    bounds=[100, 330, 900, 430],
                    text="第一行",
                    field_labels=["正文"],
                ),
            ],
            keyboard=common_keyboard,
        )
        projected = _apply_input_structure_audit(
            base,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="f" * 64,
            goal_context=context,
            ledger_input_value="第一行",
        )
        target = projected.get_element("local_audited_input_1")
        self.assertEqual("body", target.states["input_field_id"])
        self.assertEqual("正文", target.states["input_field_label"])
        self.assertEqual("第一行", target.states["value"])
        self.assertIn("正文", target.evidence)
        self.assertNotIn("主题", target.evidence)
        enter = projected.get_element("local_audited_enter_key_1")
        self.assertEqual("input_exact_enter_key", enter.meaning)
        self.assertEqual("第一行\n", enter.states["expected_input_value"])

        audit["keyboard"]["enter_key"]["key_action"] = "send"
        rejected = _apply_input_structure_audit(
            base,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="f" * 64,
            goal_context=context,
            ledger_input_value="第一行",
        )
        self.assertFalse(
            any(item.meaning == "input_exact_enter_key" for item in rejected.elements)
        )

    def test_multifield_audit_mints_typed_next_field_key_for_hidden_active_field(self) -> None:
        fields = MULTIFIELD_FIELDS
        base = multifield_next_base(fields)
        context = multifield_next_context(fields)
        audit = multifield_next_audit(fields)

        projected = _apply_input_structure_audit(
            base,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="f" * 64,
            goal_context=context,
            ledger_input_value="first",
        )

        target = projected.unique_trusted_goal_element()
        self.assertIsNotNone(target)
        self.assertEqual("input_next_field_key", target.meaning)
        self.assertEqual("subject_field", target.states["source_input_field_id"])
        self.assertEqual("body_field", target.states["target_input_field_id"])
        self.assertEqual("正文", target.states["target_input_field_label"])
        current = projected.get_element("local_audited_input_1")
        self.assertFalse(current.states["goal_relevant"])
        self.assertEqual("subject_field", current.states["input_field_id"])

    def test_multifield_next_field_key_survives_predecessor_omitted_from_coarse_scene(
        self,
    ) -> None:
        fields = MULTIFIELD_FIELDS
        base = _parse_scene(
            json.dumps(scene_payload(), ensure_ascii=False),
            fingerprint="f" * 64,
        )
        context = multifield_next_context(fields)
        audit = multifield_next_audit(fields)

        projected = _apply_input_structure_audit(
            base,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="f" * 64,
            goal_context=context,
            ledger_input_value="first",
        )

        target = projected.unique_trusted_goal_element()
        self.assertIsNotNone(target)
        self.assertEqual("input_next_field_key", target.meaning)
        self.assertEqual("subject_field", target.states["source_input_field_id"])
        self.assertEqual("body_field", target.states["target_input_field_id"])

    def test_multifield_next_field_key_rejects_compact_cue_shadow_authority(
        self,
    ) -> None:
        fields = MULTIFIELD_FIELDS
        context = multifield_next_context(fields)
        audit = multifield_next_audit(fields)
        audited_source = audit["application_inputs"][0]
        audited_source["text"] = ""
        audited_source["visible_editable_cues"] = ["first"]

        projected = _apply_input_structure_audit(
            multifield_next_base(fields),
            json.dumps(audit, ensure_ascii=False),
            fingerprint="f" * 64,
            goal_context=context,
            ledger_input_value="first",
        )

        self.assertFalse(
            any(item.meaning == "input_next_field_key" for item in projected.elements)
        )
        self.assertFalse(
            any(
                item.meaning == "application_text_input"
                and item.states.get("value") == "first"
                for item in projected.elements
            )
        )

        rejected = _apply_input_structure_audit(
            _parse_scene(
                json.dumps(scene_payload(), ensure_ascii=False),
                fingerprint="f" * 64,
            ),
            json.dumps(audit, ensure_ascii=False),
            fingerprint="f" * 64,
            goal_context=context,
            ledger_input_value="first",
        )
        self.assertFalse(
            any(item.meaning == "input_next_field_key" for item in rejected.elements)
        )

    def test_multifield_next_field_key_uses_locally_snapped_qwerty_geometry(
        self,
    ) -> None:
        fields = MULTIFIELD_FIELDS
        base = multifield_next_base(fields)
        context = multifield_next_context(fields)
        audit = multifield_next_audit(fields)
        snapped = {
            "q": [122, 709], "p": [881, 709],
            "a": [164, 781], "l": [839, 781],
            "z": [249, 853], "m": [755, 853],
            "backspace": [881, 853],
        }

        projected = _apply_input_structure_audit(
            base,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="f" * 64,
            goal_context=context,
            ledger_input_value="first",
            qwerty_row_snapper=lambda _frames, _anchors: snapped,
            qwerty_row_frames=stable_frames()[-3:],
        )

        target = projected.unique_trusted_goal_element()
        self.assertIsNotNone(target)
        self.assertEqual("input_next_field_key", target.meaning)
        self.assertEqual("next", target.states["key_action"])

        audit["keyboard"]["enter_key"]["label"] = "发送"
        rejected = _apply_input_structure_audit(
            base,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="f" * 64,
            goal_context=context,
            ledger_input_value="first",
            qwerty_row_snapper=lambda _frames, _anchors: snapped,
            qwerty_row_frames=stable_frames()[-3:],
        )
        self.assertFalse(
            any(item.meaning == "input_next_field_key" for item in rejected.elements)
        )

    def test_multifield_next_field_key_follows_typed_identity_not_array_order(self) -> None:
        cases = (
            (
                [
                    {"field_id": "subject_field", "field_label": "标题", "text": "alpha"},
                    {"field_id": "body_field", "field_label": "备注", "text": "beta"},
                ],
                "标题",
                "alpha",
                "备注",
            ),
            (
                [
                    {"field_id": "body_field", "field_label": "内容", "text": "delta"},
                    {"field_id": "subject_field", "field_label": "名称", "text": "gamma"},
                ],
                "名称",
                "gamma",
                "内容",
            ),
        )
        for fields, _source_label, source_text, target_label in cases:
            with self.subTest(fields=fields):
                base = multifield_next_base(fields)
                context = multifield_next_context(fields)
                audit = multifield_next_audit(fields)
                projected = _apply_input_structure_audit(
                    base,
                    json.dumps(audit, ensure_ascii=False),
                    fingerprint="f" * 64,
                    goal_context=context,
                    ledger_input_value=source_text,
                )
                target = projected.unique_trusted_goal_element()
                self.assertIsNotNone(target)
                self.assertEqual("body_field", target.states["target_input_field_id"])
                self.assertEqual(target_label, target.states["target_input_field_label"])

    def test_multifield_next_field_key_rejects_ambiguous_or_unbound_cases(self) -> None:
        cases = (
            ("no dependency", False, False, "next", True, 0.98),
            ("not next", True, False, "done", True, 0.98),
            ("not fully visible", True, False, "next", False, 0.98),
            ("low confidence", True, False, "next", True, 0.71),
            ("duplicate dedicated audit", True, False, "next", True, 0.98),
        )
        for name, dependency, duplicate, action, visible, confidence in cases:
            with self.subTest(name=name):
                base = multifield_next_base(MULTIFIELD_FIELDS, duplicate=duplicate)
                context = multifield_next_context(
                    MULTIFIELD_FIELDS, dependency=dependency
                )
                audit = multifield_next_audit(
                    MULTIFIELD_FIELDS, action=action,
                    fully_visible=visible, confidence=confidence,
                    duplicate=name == "duplicate dedicated audit",
                )
                projected = _apply_input_structure_audit(
                    base,
                    json.dumps(audit, ensure_ascii=False),
                    fingerprint="f" * 64,
                    goal_context=context,
                    ledger_input_value="first",
                )
                self.assertFalse(
                    any(item.meaning == "input_next_field_key" for item in projected.elements)
                )
                if action != "next":
                    self.assertNotIn(
                        "local_audited_input_1",
                        {item.element_id for item in projected.elements},
                    )

    def test_typed_prefix_survives_invalid_keyboard_only_as_verification_evidence(
        self,
    ) -> None:
        base = _parse_scene(
            json.dumps(scene_payload(), ensure_ascii=False),
            fingerprint="f" * 64,
        )
        target = "abcdefghijklmnopqrstuvwxyzabcdefghijk"
        prefix = "abcdefghijklmnopqrst"
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "enter_text",
                    "objective": "输入长文本",
                    "constraints": [],
                    "completion_conditions": [],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": target,
                        "active_input_field_id": "input_field_1",
                        "active_input_field_label": "长文本",
                        "active_input_multiline": False,
                    },
                }
            }
        }
        structure = audited_application_input(
            bounds=[130, 590, 870, 740],
            text=prefix,
            field_labels=["长文本"],
        )
        structure["visible_editable_cues"] = ["cursor", "border"]
        invalid_keyboard = {
            "visible": True,
            "bounds": [0, 830, 1000, 1000],
            "layout": "qwerty",
            "input_mode": "direct_latin",
            "case_mode": "lower",
            "qwerty_anchors": {
                "q": [110, 910],
                "p": [890, 910],
                "a": [160, 950],
                "l": [840, 950],
                "z": [260, 990],
                "m": [740, 990],
                "backspace": [890, 990],
            },
            "mode_switch": None,
            "backspace_key": None,
            "enter_key": None,
            "case_switch": None,
            "literal_keys": [],
            "layout_switches": [],
        }
        audit = input_audit_payload(
            application_inputs=[structure],
            keyboard=invalid_keyboard,
        )

        projected = _apply_input_structure_audit(
            base,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="f" * 64,
            goal_context=context,
            ledger_input_value=prefix,
        )

        field = projected.get_element("local_audited_input_1")
        self.assertEqual(prefix, field.states["value"])
        self.assertEqual("input_field_1", field.states["input_field_id"])
        self.assertFalse(field.states["goal_relevant"])
        self.assertNotIn("keyboard_geometry", field.states)
        self.assertNotIn("focused", field.states)

        for invalid_inputs in (
            [{**structure, "text": "wrong-prefix"}],
            [structure, {**structure, "structure_id": "other-input"}],
            [{**structure, "text": target}],
        ):
            with self.subTest(invalid_inputs=invalid_inputs), self.assertRaisesRegex(
                VisionAgentError,
                "可见键盘必须提供有效 bounds",
            ):
                _apply_input_structure_audit(
                    base,
                    json.dumps(
                        input_audit_payload(
                            application_inputs=invalid_inputs,
                            keyboard=invalid_keyboard,
                        ),
                        ensure_ascii=False,
                    ),
                    fingerprint="f" * 64,
                    goal_context=context,
                    ledger_input_value=prefix,
                )

    def test_local_ocr_rows_repair_compressed_qwerty_before_input_authorization(
        self,
    ) -> None:
        base = _parse_scene(
            json.dumps(scene_payload(), ensure_ascii=False),
            fingerprint="f" * 64,
        )
        context = {
            "objective": "在唯一输入框输入 agent",
            "entities": {"input_text": "agent"},
        }
        audit = input_audit_payload(
            application_inputs=[audited_application_input(text="", placeholder="输入")],
            keyboard={
                "visible": True,
                "bounds": [50, 830, 950, 1000],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "case_mode": "lower",
                "qwerty_anchors": {
                    "q": [110, 895],
                    "p": [890, 895],
                    "a": [160, 945],
                    "l": [840, 945],
                    "z": [260, 990],
                    "m": [740, 990],
                    "backspace": [890, 990],
                },
                "mode_switch": None,
            },
        )
        snapped = {
            "q": [110, 704],
            "p": [890, 704],
            "a": [160, 774],
            "l": [840, 774],
            "z": [260, 844],
            "m": [740, 844],
            "backspace": [890, 844],
        }

        projected = _apply_input_structure_audit(
            base,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="f" * 64,
            goal_context=context,
            ledger_input_value="",
            qwerty_row_snapper=lambda _frames, _anchors: snapped,
            qwerty_row_frames=stable_frames()[-3:],
        )

        geometry = projected.get_element(
            "local_audited_input_1"
        ).states["keyboard_geometry"]
        self.assertEqual("input_structure_audit", geometry["source"])
        self.assertEqual(snapped, geometry["anchors"])

    def test_locally_snapped_qwerty_anchors_replace_missing_outer_bounds(
        self,
    ) -> None:
        base = _parse_scene(
            json.dumps(scene_payload(), ensure_ascii=False),
            fingerprint="f" * 64,
        )
        context = {
            "objective": "在唯一输入框输入 agent",
            "entities": {"input_text": "agent"},
        }
        for snapped in (
            {
                "q": [110, 704],
                "p": [890, 704],
                "a": [160, 774],
                "l": [840, 774],
                "z": [260, 844],
                "m": [740, 844],
                "backspace": [890, 844],
            },
            {
                "q": [120, 620],
                "p": [880, 620],
                "a": [165, 700],
                "l": [835, 700],
                "z": [255, 780],
                "m": [745, 780],
                "backspace": [890, 780],
            },
        ):
            with self.subTest(snapped=snapped):
                audit = input_audit_payload(
                    application_inputs=[
                        audited_application_input(text="", placeholder="输入")
                    ],
                    keyboard={
                        "visible": True,
                        "bounds": None,
                        "layout": "qwerty",
                        "input_mode": "direct_latin",
                        "case_mode": "lower",
                        "qwerty_anchors": snapped,
                        "mode_switch": None,
                    },
                )

                projected = _apply_input_structure_audit(
                    base,
                    json.dumps(audit, ensure_ascii=False),
                    fingerprint="f" * 64,
                    goal_context=context,
                    ledger_input_value="",
                    qwerty_row_snapper=lambda _frames, _anchors: snapped,
                    qwerty_row_frames=stable_frames()[-3:],
                )

                geometry = projected.get_element(
                    "local_audited_input_1"
                ).states["keyboard_geometry"]
                self.assertEqual("input_structure_audit", geometry["source"])
                self.assertEqual(snapped, geometry["anchors"])

    def test_anchor_derived_bounds_keep_newline_key_independently_audited(
        self,
    ) -> None:
        base = _parse_scene(
            json.dumps(scene_payload(), ensure_ascii=False),
            fingerprint="f" * 64,
        )
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "fill_body",
                    "objective": "正文内容逐字等于目标文本",
                    "constraints": [],
                    "completion_conditions": [],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "first line\nsecond line",
                        "active_input_field_id": "body",
                        "active_input_field_label": "正文",
                        "active_input_multiline": True,
                    },
                }
            }
        }
        snapped = {
            "q": [110, 708],
            "p": [890, 708],
            "a": [150, 780],
            "l": [850, 780],
            "z": [260, 852],
            "m": [740, 852],
            "backspace": [890, 852],
        }
        keyboard = {
            "visible": True,
            "bounds": None,
            "layout": "qwerty",
            "input_mode": "direct_latin",
            "case_mode": "lower",
            "qwerty_anchors": snapped,
            "mode_switch": None,
            "enter_key": {
                "label": "↵",
                "bounds": [820, 900, 950, 970],
                "confidence": 0.98,
                "fully_visible": True,
                "key_action": "newline",
            },
        }
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="body",
                    bounds=[100, 330, 900, 430],
                    text="first line",
                    field_labels=["正文"],
                )
            ],
            keyboard=keyboard,
        )

        projected = _apply_input_structure_audit(
            base,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="f" * 64,
            goal_context=context,
            ledger_input_value="first line",
            qwerty_row_snapper=lambda _frames, _anchors: snapped,
            qwerty_row_frames=stable_frames()[-3:],
        )

        enter = projected.get_element("local_audited_enter_key_1")
        self.assertEqual("input_exact_enter_key", enter.meaning)
        self.assertEqual("first line\n", enter.states["expected_input_value"])

        audit["keyboard"]["enter_key"]["fully_visible"] = False
        rejected = _apply_input_structure_audit(
            base,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="f" * 64,
            goal_context=context,
            ledger_input_value="first line",
            qwerty_row_snapper=lambda _frames, _anchors: snapped,
            qwerty_row_frames=stable_frames()[-3:],
        )
        self.assertFalse(
            any(item.meaning == "input_exact_enter_key" for item in rejected.elements)
        )

    def test_local_rows_snap_newline_key_from_untrusted_portrait_grid(self) -> None:
        coarse = scene_payload()
        coarse["elements"].append(
            {
                "element_id": "coarse-body",
                "role": "input",
                "meaning": "application_text_input",
                "label": "正文",
                "bounds": [100, 250, 900, 450],
                "confidence": 0.98,
                "states": {
                    "goal_relevant": True,
                    "fully_visible": True,
                    "placeholder": "正文",
                },
                "evidence": ["正文"],
            }
        )
        base = _parse_scene(
            json.dumps(coarse, ensure_ascii=False),
            fingerprint="f" * 64,
        )
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "fill_body",
                    "objective": "正文内容逐字等于目标文本",
                    "constraints": [],
                    "completion_conditions": [],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "first\nsecond",
                        "active_input_field_id": "body",
                        "active_input_field_label": "正文",
                        "active_input_multiline": True,
                    },
                }
            }
        }
        raw_keyboard = {
            "visible": True,
            "bounds": [60, 1160, 940, 1900],
            "layout": "qwerty",
            "input_mode": "direct_latin",
            "case_mode": "lower",
            "qwerty_anchors": {
                "q": [110, 1430],
                "p": [890, 1430],
                "a": [160, 1580],
                "l": [840, 1580],
                "z": [260, 1730],
                "m": [740, 1730],
                "backspace": [890, 1730],
            },
            "mode_switch": None,
            "enter_key": {
                "label": "",
                "bounds": [840, 1830, 930, 1890],
                "confidence": 0.98,
                "fully_visible": True,
                "key_action": "newline",
            },
        }
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="body",
                    bounds=[100, 330, 900, 430],
                    text="first",
                    field_labels=["正文"],
                )
            ],
            keyboard=raw_keyboard,
        )

        for snapped in (
            {
                "q": [122, 708],
                "p": [881, 708],
                "a": [164, 780],
                "l": [839, 780],
                "z": [249, 852],
                "m": [755, 852],
                "backspace": [881, 852],
            },
            {
                "q": [122, 620],
                "p": [881, 620],
                "a": [164, 700],
                "l": [839, 700],
                "z": [249, 780],
                "m": [755, 780],
                "backspace": [881, 780],
            },
        ):
            with self.subTest(snapped=snapped):
                projected = _apply_input_structure_audit(
                    base,
                    json.dumps(audit, ensure_ascii=False),
                    fingerprint="f" * 64,
                    goal_context=context,
                    ledger_input_value="first",
                    qwerty_row_snapper=lambda _frames, _anchors: snapped,
                    qwerty_row_frames=stable_frames()[-3:],
                )
                enter = projected.get_element("local_audited_enter_key_1")
                field = projected.get_element("local_audited_input_1")
                self.assertEqual((0.1, 0.25, 0.9, 0.45), field.bounds)
                expected_center_y = (
                    snapped["z"][1]
                    + (snapped["a"][1] - snapped["q"][1])
                ) / 1000.0
                self.assertAlmostEqual(
                    expected_center_y,
                    (enter.bounds[1] + enter.bounds[3]) / 2.0,
                    places=3,
                )
                self.assertEqual("first\n", enter.states["expected_input_value"])
                self.assertEqual("↵", enter.label)

        audit["keyboard"]["enter_key"]["key_action"] = "next"
        rejected_unlabeled_next = _apply_input_structure_audit(
            base,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="f" * 64,
            goal_context=context,
            ledger_input_value="first",
            qwerty_row_snapper=lambda _frames, _anchors: {
                "q": [122, 708],
                "p": [881, 708],
                "a": [164, 780],
                "l": [839, 780],
                "z": [249, 852],
                "m": [755, 852],
                "backspace": [881, 852],
            },
            qwerty_row_frames=stable_frames()[-3:],
        )
        self.assertFalse(
            any(
                item.meaning == "input_exact_enter_key"
                for item in rejected_unlabeled_next.elements
            )
        )

        audit["keyboard"]["enter_key"]["key_action"] = "newline"
        audit["keyboard"]["enter_key"]["label"] = "开始"
        rejected = _apply_input_structure_audit(
            base,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="f" * 64,
            goal_context=context,
            ledger_input_value="first",
            qwerty_row_snapper=lambda _frames, _anchors: {
                "q": [122, 708],
                "p": [881, 708],
                "a": [164, 780],
                "l": [839, 780],
                "z": [249, 852],
                "m": [755, 852],
                "backspace": [881, 852],
            },
            qwerty_row_frames=stable_frames()[-3:],
        )
        self.assertFalse(
            any(item.meaning == "input_exact_enter_key" for item in rejected.elements)
        )

    def test_local_rows_do_not_reattach_ambiguous_scene_fields(self) -> None:
        coarse = scene_payload()
        for index, top in enumerate((180, 330), start=1):
            coarse["elements"].append(
                {
                    "element_id": f"coarse-body-{index}",
                    "role": "input",
                    "meaning": "application_text_input",
                    "label": "正文",
                    "bounds": [100, top, 900, top + 100],
                    "confidence": 0.98,
                    "states": {"goal_relevant": True, "fully_visible": True},
                    "evidence": ["正文"],
                }
            )
        base = _parse_scene(
            json.dumps(coarse, ensure_ascii=False),
            fingerprint="f" * 64,
        )
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "fill_body",
                    "objective": "正文内容逐字等于目标文本",
                    "constraints": [],
                    "completion_conditions": [],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "first\nsecond",
                        "active_input_field_id": "body",
                        "active_input_field_label": "正文",
                        "active_input_multiline": True,
                    },
                }
            }
        }
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="body",
                    bounds=[135, 490, 865, 730],
                    text="first",
                    field_labels=["正文"],
                )
            ],
            keyboard={
                "visible": True,
                "bounds": [50, 870, 950, 1000],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "case_mode": "lower",
                "qwerty_anchors": {
                    "q": [100, 935],
                    "p": [890, 935],
                    "a": [140, 965],
                    "l": [820, 965],
                    "z": [250, 990],
                    "m": [740, 990],
                    "backspace": [890, 990],
                },
                "mode_switch": None,
                "enter_key": {
                    "label": "↵",
                    "bounds": [890, 990, 950, 1000],
                    "confidence": 0.98,
                    "fully_visible": True,
                    "key_action": "newline",
                },
            },
        )
        snapped = {
            "q": [122, 708],
            "p": [881, 708],
            "a": [164, 780],
            "l": [839, 780],
            "z": [249, 852],
            "m": [755, 852],
            "backspace": [881, 852],
        }

        projected = _apply_input_structure_audit(
            base,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="f" * 64,
            goal_context=context,
            ledger_input_value="first",
            qwerty_row_snapper=lambda _frames, _anchors: snapped,
            qwerty_row_frames=stable_frames()[-3:],
        )

        field = projected.get_element("local_audited_input_1")
        self.assertEqual((0.135, 0.49, 0.865, 0.73), field.bounds)

    def test_invalid_keyboard_bounds_still_fail_without_local_row_evidence(
        self,
    ) -> None:
        base = _parse_scene(
            json.dumps(scene_payload(), ensure_ascii=False),
            fingerprint="f" * 64,
        )
        audit = input_audit_payload(
            application_inputs=[audited_application_input(text="", placeholder="输入")],
            keyboard={
                "visible": True,
                "bounds": [50, 830, 950, 1000],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "qwerty_anchors": {
                    "q": [110, 895],
                    "p": [890, 895],
                    "a": [160, 945],
                    "l": [840, 945],
                    "z": [260, 990],
                    "m": [740, 990],
                    "backspace": [890, 990],
                },
                "mode_switch": None,
            },
        )

        with self.assertRaisesRegex(VisionAgentError, "可见键盘必须提供有效 bounds"):
            _apply_input_structure_audit(
                base,
                json.dumps(audit, ensure_ascii=False),
                fingerprint="f" * 64,
                goal_context={
                    "objective": "在唯一输入框输入 agent",
                    "entities": {"input_text": "agent"},
                },
                ledger_input_value="",
                qwerty_row_snapper=lambda _frames, _anchors: None,
                qwerty_row_frames=stable_frames()[-3:],
            )

    def test_unique_typed_active_field_accepts_tall_input_without_minting_multiline(
        self,
    ) -> None:
        base = _parse_scene(
            json.dumps(scene_payload(), ensure_ascii=False),
            fingerprint="f" * 64,
        )
        cases = [
            (MULTIFIELD_FIELDS, [130, 630, 870, 830]),
            (
                [
                    {"field_id": "body_field", "field_label": "详细内容", "text": "second"},
                    {"field_id": "subject_field", "field_label": "标题内容", "text": "first"},
                ],
                [130, 300, 870, 850],
            ),
        ]
        for fields, body_bounds in cases:
            with self.subTest(fields=fields, body_bounds=body_bounds):
                source = next(item for item in fields if item["field_id"] == "subject_field")
                target = next(item for item in fields if item["field_id"] == "body_field")
                inputs_by_id = {
                    "subject_field": audited_application_input(
                        structure_id="subject",
                        bounds=[130, 150, 870, 250],
                        text=source["text"],
                        field_labels=[source["field_label"]],
                    ),
                    "body_field": audited_application_input(
                        structure_id="body",
                        bounds=body_bounds,
                        text="",
                        placeholder=target["field_label"],
                        field_labels=[target["field_label"]],
                    ),
                }
                raw = json.dumps(
                    input_audit_payload(
                        application_inputs=[inputs_by_id[item["field_id"]] for item in fields]
                    ),
                    ensure_ascii=False,
                )

                projected = _apply_input_structure_audit(
                    base,
                    raw,
                    fingerprint="f" * 64,
                    goal_context=multifield_next_context(fields),
                    ledger_input_value="",
                )

                field = projected.get_element("local_audited_input_1")
                self.assertEqual(target["field_id"], field.states["input_field_id"])
                self.assertEqual(target["field_label"], field.states["input_field_label"])
                self.assertEqual("", field.states["value"])
                self.assertFalse(field.states["input_multiline"])
                self.assertFalse(field.states["soft_keyboard_visible"])

    def test_tall_single_line_typed_field_override_remains_fail_closed(self) -> None:
        base = _parse_scene(
            json.dumps(scene_payload(), ensure_ascii=False),
            fingerprint="f" * 64,
        )
        subject = audited_application_input(
            structure_id="subject", bounds=[130, 490, 870, 590],
            text="first", field_labels=["主题"],
        )
        body = audited_application_input(
            structure_id="body", bounds=[130, 630, 870, 830],
            text="", placeholder="正文", field_labels=["正文"],
        )
        cases: list[tuple[str, list[dict], dict | None, list[dict] | None]] = []
        cases.append(("duplicate-label", [subject, body, {
            **body, "structure_id": "body-duplicate", "bounds": [130, 260, 870, 460],
        }], None, None))
        for name, changes in (
            ("no-label", {"field_labels": []}),
            ("low-confidence", {"confidence": 0.7}),
            ("incomplete", {"fully_visible": False}),
            ("no-editable-evidence", {"placeholder": "", "visible_editable_cues": []}),
            ("over-maximum-height", {"bounds": [130, 100, 870, 750]}),
        ):
            cases.append((name, [subject, {**body, **changes}], None, None))
        cases.append((
            "keyboard-overlap", [subject, body],
            {
                "visible": True, "bounds": [0, 600, 1000, 1000],
                "layout": "qwerty", "input_mode": "direct_latin",
                "case_mode": "lower", "mode_switch": None,
            }, None,
        ))
        cases.append((
            "preedit-overlap", [subject, body], None,
            [{
                "region_id": "preedit", "bounds": [130, 630, 870, 830],
                "text": "second", "confidence": 0.98, "candidates": [],
            }],
        ))

        for name, inputs, keyboard, preedits in cases:
            with self.subTest(name=name):
                rejected = _apply_input_structure_audit(
                    base,
                    json.dumps(input_audit_payload(
                        application_inputs=inputs,
                        keyboard=keyboard,
                        ime_preedit_regions=preedits,
                    ), ensure_ascii=False),
                    fingerprint="f" * 64,
                    goal_context=multifield_next_context(MULTIFIELD_FIELDS),
                    ledger_input_value="",
                )
                self.assertFalse(any(
                    item.element_id == "local_audited_input_1"
                    for item in rejected.elements
                ))

        untyped = _apply_input_structure_audit(
            base,
            json.dumps(input_audit_payload(
                application_inputs=[body],
            ), ensure_ascii=False),
            fingerprint="f" * 64,
            goal_context={"objective": "输入second", "entities": {"input_text": "second"}},
            ledger_input_value="",
        )
        self.assertFalse(any(
            item.element_id == "local_audited_input_1"
            for item in untyped.elements
        ))

        duplicate_typed_label = json.loads(json.dumps(MULTIFIELD_FIELDS))
        duplicate_typed_label[0]["field_label"] = "正文"
        ambiguous_typed = _apply_input_structure_audit(
            base,
            json.dumps(input_audit_payload(
                application_inputs=[subject, body],
            ), ensure_ascii=False),
            fingerprint="f" * 64,
            goal_context=multifield_next_context(duplicate_typed_label),
            ledger_input_value="",
        )
        self.assertFalse(any(
            item.element_id == "local_audited_input_1"
            for item in ambiguous_typed.elements
        ))

        invalid = {**body, "bounds": [130, 630, 870, 1001]}
        with self.assertRaisesRegex(VisionAgentError, "bounds"):
            _apply_input_structure_audit(
                base,
                json.dumps(input_audit_payload(
                    application_inputs=[subject, invalid],
                ), ensure_ascii=False),
                fingerprint="f" * 64,
                goal_context=multifield_next_context(MULTIFIELD_FIELDS),
                ledger_input_value="",
            )

    def test_fused_unique_empty_input_needs_same_value_and_overlapping_geometry(
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
                        "input_text": "aaazjie？你好",
                        "active_input_transaction_text": "aaazjie？你好",
                        "active_input_field_id": "input_field_1",
                        "active_input_multiline": False,
                    },
                }
            }
        }
        preliminary = scene_payload()
        preliminary["elements"] = [
            {
                "element_id": "e2",
                "role": "input",
                "meaning": "message_input_field",
                "label": "",
                "bounds": [120, 910, 780, 960],
                "confidence": 1.0,
                "states": {
                    "goal_relevant": True,
                    "fully_visible": True,
                    "value": "",
                },
                "evidence": ["底部工具栏中唯一完整白色文本输入区域"],
            }
        ]
        attestation = _fused_preliminary_input_attestation(
            preliminary,
            goal_context=context,
        )
        self.assertIsNotNone(attestation)

        base_payload = scene_payload()
        base_payload["elements"] = []
        base = _parse_scene(
            json.dumps(base_payload, ensure_ascii=False),
            fingerprint="f" * 64,
        )
        audit_input = audited_application_input(
            bounds=[120, 910, 780, 960],
            text="",
            placeholder="",
            visible_editable_cues=[],
        )
        raw = json.dumps(
            input_audit_payload(application_inputs=[audit_input]),
            ensure_ascii=False,
        )

        projected = _apply_input_structure_audit(
            base,
            raw,
            fingerprint="f" * 64,
            goal_context=context,
            ledger_input_value="",
            fused_input_attestation=attestation,
        )

        field = projected.get_element("local_audited_input_1")
        self.assertEqual("", field.states["value"])
        self.assertTrue(field.states["goal_relevant"])
        self.assertIn("当前输入框为空", " ".join(field.evidence))

        for invalid_attestation in (
            None,
            {**attestation, "value": "旧值"},
            {**attestation, "bounds": [120, 700, 780, 750]},
        ):
            with self.subTest(attestation=invalid_attestation):
                rejected = _apply_input_structure_audit(
                    base,
                    raw,
                    fingerprint="f" * 64,
                    goal_context=context,
                    ledger_input_value="",
                    fused_input_attestation=invalid_attestation,
                )
                self.assertFalse(any(
                    item.element_id == "local_audited_input_1"
                    for item in rejected.elements
                ))

    def test_typed_multiline_prefix_survives_placeholder_loss_and_pixel_coordinates(
        self,
    ) -> None:
        base = _parse_scene(
            json.dumps(scene_payload(), ensure_ascii=False),
            fingerprint="f" * 64,
        )
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_exact_text",
                    "objective": "输入两行文字",
                    "constraints": ["不要提交"],
                    "completion_conditions": ["输入框逐字等于授权文字"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "input_text": "first\nsecond",
                        "active_input_transaction_text": "first\nsecond",
                        "active_input_field_id": "input_field_1",
                        "active_input_multiline": True,
                    },
                }
            }
        }
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    bounds=[135, 490, 860, 810],
                    text="first",
                    placeholder="",
                    field_labels=["正文"],
                    caret_line_index=1,
                )
            ],
            keyboard={
                "visible": True,
                "bounds": [60, 1080, 940, 1980],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "case_mode": "lower",
                "qwerty_anchors": {
                    "q": [110, 1380],
                    "p": [890, 1380],
                    "a": [160, 1560],
                    "l": [840, 1560],
                    "z": [260, 1740],
                    "m": [740, 1740],
                    "backspace": [890, 1740],
                },
                "mode_switch": None,
                "enter_key": {
                    "label": "↵",
                    "bounds": [840, 1840, 940, 1940],
                    "confidence": 1.0,
                    "fully_visible": True,
                    "key_action": "newline",
                },
            },
        )
        audit["application_inputs"][0]["visible_editable_cues"] = [
            "border",
            "cursor",
        ]
        frames = tuple(
            Image.new("RGB", (1000, 2000), "white") for _ in range(4)
        )
        newline_lineage = TypedInputLineage(
            version=TYPED_INPUT_LINEAGE_VERSION,
            device_id="device-local-01",
            exact_value="first\n",
            app_id="calculator",
            screen_id="app_home",
            input_meaning="application_text_input",
            input_field_id="input_field_1",
            input_bounds=(0.135, 0.245, 0.86, 0.405),
            before_fingerprint="before-newline",
            after_fingerprint="pending-visual-verification",
            action_digest="c" * 64,
            receipt_digest="d" * 64,
            surface_descriptors=(),
            recorded_at_epoch=time.time(),
            source="pending_verified_newline_action",
        )
        newline_lineage.validate()

        projected = _apply_input_structure_audit(
            base,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="f" * 64,
            goal_context=context,
            ledger_input_value="conflicting compact value",
            verified_input_lineage=newline_lineage,
            device_id="device-local-01",
            qwerty_row_frames=frames,
        )

        field = projected.get_element("local_audited_input_1")
        self.assertEqual("input_field_1", field.states["input_field_id"])
        self.assertEqual("first\n", field.states["value"])
        self.assertTrue(field.states["goal_relevant"])
        self.assertEqual("qwerty", field.states["keyboard_geometry"]["type"])
        self.assertEqual([110, 690], field.states["keyboard_geometry"]["anchors"]["q"])

    def test_typed_field_recovers_authorized_committed_prefix_without_screen_lineage(
        self,
    ) -> None:
        base_payload = scene_payload()
        base_payload["screen_id"] = "chat_input"
        base_payload["elements"] = []
        base = _parse_scene(
            json.dumps(base_payload, ensure_ascii=False),
            fingerprint="authorized-prefix-frame",
        )
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_exact_text",
                    "objective": "输入两行文字",
                    "constraints": ["不要提交"],
                    "completion_conditions": ["输入框逐字等于授权文字"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "input_text": "first\nsecond",
                        "active_input_transaction_text": "first\nsecond",
                        "active_input_field_id": "input_field_1",
                        "active_input_multiline": True,
                    },
                }
            }
        }
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="message-field",
                    bounds=[140, 540, 720, 630],
                    text="",
                    placeholder="",
                )
            ],
            ime_preedit_regions=[],
            keyboard={
                "visible": True,
                "bounds": [0, 660, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "case_mode": "lower",
                "mode_switch": None,
                "enter_key": {
                    "label": "↵",
                    "bounds": [840, 880, 960, 960],
                    "confidence": 1.0,
                    "fully_visible": True,
                    "key_action": "newline",
                },
            },
        )
        audit["application_inputs"][0]["visible_editable_cues"] = ["first"]

        projected = _apply_input_structure_audit(
            base,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="authorized-prefix-frame",
            goal_context=context,
            ledger_input_value="",
        )

        field = projected.get_element("local_audited_input_1")
        self.assertEqual("first", field.states["value"])
        self.assertEqual("input_field_1", field.states["input_field_id"])
        enter = projected.get_element("local_audited_enter_key_1")
        self.assertEqual("first\n", enter.states["expected_input_value"])
        self.assertTrue(
            any("授权payload前缀" in item for item in field.evidence)
        )

        candidate_less_direct_region = json.loads(json.dumps(audit))
        candidate_less_direct_region["ime_preedit_regions"] = [
            {
                "region_id": "misclassified-direct-region",
                "bounds": [140, 540, 300, 600],
                "text": "first",
                "confidence": 1.0,
                "candidates": [],
            }
        ]
        recovered_candidate_less_direct_region = _apply_input_structure_audit(
            base,
            json.dumps(candidate_less_direct_region, ensure_ascii=False),
            fingerprint="authorized-prefix-direct-region",
            goal_context=context,
            ledger_input_value="",
        )
        self.assertEqual(
            "first",
            recovered_candidate_less_direct_region.get_element(
                "local_audited_input_1"
            ).states["value"],
        )

        variations = []
        wrong_cue = json.loads(json.dumps(audit))
        wrong_cue["application_inputs"][0]["visible_editable_cues"] = ["firstly"]
        variations.append(("wrong_cue", context, wrong_cue))
        placeholder = json.loads(json.dumps(audit))
        placeholder["application_inputs"][0]["placeholder"] = "first"
        variations.append(("placeholder", context, placeholder))
        preedit = json.loads(json.dumps(audit))
        preedit["ime_preedit_regions"] = [
            {
                "region_id": "preedit-1",
                "bounds": [140, 540, 300, 600],
                "text": "first",
                "confidence": 1.0,
                "candidates": [
                    {
                        "text": "first",
                        "bounds": [140, 630, 260, 660],
                        "confidence": 1.0,
                        "fully_visible": True,
                    }
                ],
            }
        ]
        variations.append(("preedit", context, preedit))
        missing_field = json.loads(json.dumps(context))
        missing_field["entities"]["active_subgoal_visual_context"][
            "goal_entities"
        ].pop("active_input_field_id")
        variations.append(("missing_field", missing_field, audit))
        for name, candidate_context, candidate_audit in variations:
            with self.subTest(name=name):
                rejected = _apply_input_structure_audit(
                    base,
                    json.dumps(candidate_audit, ensure_ascii=False),
                    fingerprint=f"authorized-prefix-rejected-{name}",
                    goal_context=candidate_context,
                    ledger_input_value="",
                )
                self.assertNotEqual(
                    "first",
                    rejected.get_element("local_audited_input_1").states["value"],
                )

    def test_newline_locator_can_recover_tall_multiline_field_without_action_identity(
        self,
    ) -> None:
        base = _parse_scene(
            json.dumps(scene_payload(), ensure_ascii=False),
            fingerprint="f" * 64,
        )
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    bounds=[130, 240, 870, 590],
                    text="",
                    placeholder="正文",
                )
            ]
        )
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "locate_input",
                    "objective": "定位唯一多行输入框",
                    "constraints": ["不得点击或输入"],
                    "completion_conditions": ["多行输入框可见"],
                    "execution_class": "observe",
                    "goal_entities": {
                        "input_text": "first line\nsecond line",
                        "target_ui_label": "多行输入框",
                    },
                }
            }
        }

        projected = _apply_input_structure_audit(
            base,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="f" * 64,
            goal_context=context,
            ledger_input_value="",
        )

        field = projected.get_element("local_audited_input_1")
        self.assertNotIn("input_field_id", field.states)
        self.assertEqual((0.13, 0.24, 0.87, 0.59), field.bounds)

    def test_direct_segment_discards_invalid_unused_keyboard_controls(self) -> None:
        base = _parse_scene(
            json.dumps(scene_payload(), ensure_ascii=False),
            fingerprint="f" * 64,
        )
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    bounds=[135, 490, 860, 730],
                    text="",
                    placeholder="正文",
                    field_labels=["正文"],
                )
            ],
            keyboard={
                "visible": True,
                "bounds": [60, 830, 990, 1000],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "case_mode": "lower",
                "qwerty_anchors": {
                    "q": [110, 915],
                    "p": [880, 915],
                    "a": [150, 955],
                    "l": [810, 955],
                    "z": [250, 990],
                    "m": [730, 990],
                    "backspace": [900, 990],
                },
                "mode_switch": {
                    "label": "英",
                    "bounds": [710, 1160, 790, 1200],
                    "confidence": 1.0,
                    "current_mode": "direct_latin",
                    "target_mode": "chinese_pinyin",
                },
                "backspace_key": {
                    "label": "",
                    "bounds": [860, 1140, 960, 1200],
                    "confidence": 1.0,
                    "fully_visible": True,
                },
                "enter_key": {
                    "label": "↵",
                    "bounds": [860, 1210, 960, 1270],
                    "confidence": 1.0,
                    "fully_visible": True,
                    "key_action": "newline",
                },
                "case_switch": {
                    "label": "⇧",
                    "bounds": [60, 1140, 160, 1200],
                    "confidence": 1.0,
                    "current_mode": "lower",
                    "target_mode": "upper",
                },
                "literal_keys": [],
                "layout_switches": [
                    {
                        "label": "123",
                        "bounds": [210, 1210, 310, 1270],
                        "confidence": 1.0,
                        "current_layout": "qwerty",
                        "target_layout": "numeric",
                    }
                ],
            },
        )
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "enter_text",
                    "objective": "输入两行文本",
                    "constraints": [],
                    "completion_conditions": [],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "input_text": "first line\nsecond line",
                        "active_input_transaction_text": "first line\nsecond line",
                        "active_input_field_id": "input_field_1",
                        "active_input_multiline": True,
                    },
                }
            }
        }
        snapped = {
            "q": [110, 708],
            "p": [880, 708],
            "a": [150, 780],
            "l": [810, 780],
            "z": [250, 853],
            "m": [730, 853],
            "backspace": [900, 853],
        }

        projected = _apply_input_structure_audit(
            base,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="f" * 64,
            goal_context=context,
            ledger_input_value="",
            qwerty_row_snapper=lambda _frames, _anchors: snapped,
            qwerty_row_frames=stable_frames()[-3:],
        )

        field = projected.get_element("local_audited_input_1")
        self.assertTrue(field.states["goal_relevant"])
        self.assertEqual(snapped, field.states["keyboard_geometry"]["anchors"])
        self.assertFalse(
            any(item.element_id.startswith("local_audited_keyboard_") for item in projected.elements)
        )

    def test_input_audit_rejects_protocol_external_fields(self) -> None:
        empty = scene_payload()
        empty["elements"] = []
        provider = SequenceProvider(
            [
                empty,
                empty,
                {**input_audit_payload(), "next_action": "tap"},
            ]
        )

        with self.assertRaisesRegex(VisionAgentError, "协议外字段"):
            GenericSceneObserver(provider).observe(
                frames=stable_frames(),
                goal_context={"objective": "修改顶部搜索输入框中的文字"},
            )

    def test_adjacent_submit_and_internal_icon_still_yield_only_input_candidate(self) -> None:
        payload = scene_payload()
        payload["elements"] = [
            {
                "element_id": "field",
                "role": "container",
                "meaning": "query_input_container",
                "label": "查询区域",
                "bounds": [110, 80, 740, 180],
                "confidence": 0.98,
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": ["横向边框"],
            },
            {
                "element_id": "query",
                "role": "text",
                "meaning": "current_query_text",
                "label": "已有文字",
                "bounds": [200, 105, 475, 155],
                "confidence": 0.99,
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": ["已有文字"],
            },
            {
                "element_id": "decoration",
                "role": "icon",
                "meaning": "field_leading_icon",
                "label": "装饰图标",
                "bounds": [130, 105, 180, 155],
                "confidence": 0.99,
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": ["输入区内图标"],
            },
            {
                "element_id": "submit",
                "role": "button",
                "meaning": "search_action_button",
                "label": "搜索",
                "bounds": [700, 80, 830, 180],
                "confidence": 0.97,
                "states": {"goal_relevant": False, "fully_visible": True},
                "evidence": ["右侧独立按钮"],
            },
        ]
        observer = GenericSceneObserver(FakeProvider(payload))

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={"objective": "修改顶部搜索输入框中的文字"},
        )

        candidate = scene.unique_trusted_goal_element()
        self.assertIsNotNone(candidate)
        self.assertEqual("local_structured_input_1", candidate.element_id)
        self.assertEqual((0.11, 0.08, 0.7, 0.18), candidate.bounds)
        self.assertFalse(scene.get_element("decoration").states["goal_relevant"])
        self.assertFalse(scene.get_element("submit").states["goal_relevant"])

    def test_target_app_already_open_does_not_refine_open_goal(self) -> None:
        payload = scene_payload()
        payload["foreground_app_id"] = "wechat"
        payload["screen_id"] = "chat_list"
        provider = SequenceProvider([payload])
        observer = GenericSceneObserver(provider)
        observer.observe(
            frames=stable_frames(),
            goal_context={
                "app_id": "wechat",
                "app_name": "微信",
                "objective": "打开微信",
            },
        )
        self.assertEqual(provider.calls, 1)
        self.assertFalse(observer.last_diagnostics["targeted_refinement_used"])

    def test_title_goal_refines_and_promotes_unique_page_title_identity(self) -> None:
        compact = scene_payload()
        compact["screen_id"] = "unknown"
        compact["elements"] = [
            {
                "element_id": "return-link",
                "role": "button",
                "meaning": "return_to_previous",
                "label": "返回",
                "bounds": [100, 230, 300, 280],
                "confidence": 1.0,
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": ["页面上方返回文字清晰可见"],
            }
        ]
        refined = json.loads(json.dumps(compact, ensure_ascii=False))
        refined["elements"] = [
            {
                "element_id": "page-title",
                "role": "text",
                "meaning": "page_title",
                "label": "通用动作真机验收页",
                "bounds": [100, 100, 700, 180],
                "confidence": 1.0,
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": ["页面顶部唯一大号主标题逐字清晰可见"],
            }
        ]
        provider = SequenceProvider([compact, refined])
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={"objective": "看清下一页标题"},
        )

        self.assertEqual(2, provider.calls)
        self.assertTrue(observer.last_diagnostics["targeted_refinement_used"])
        self.assertEqual("unknown", scene.screen_id)
        self.assertEqual("page_title", scene.elements[0].meaning)
        self.assertEqual("通用动作真机验收页", scene.elements[0].label)

    def test_high_confidence_goal_element_does_not_refine_only_for_unknown_screen(self) -> None:
        payload = scene_payload()
        payload["foreground_app_id"] = "unknown"
        payload["screen_id"] = "unknown"
        payload["elements"][0]["states"] = {"goal_relevant": True}
        provider = SequenceProvider([payload])
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={
                "app_id": "current_foreground",
                "objective": "让目标进入当前画面",
            },
        )

        self.assertEqual(provider.calls, 1)
        self.assertEqual("unknown", scene.screen_id)
        self.assertFalse(observer.last_diagnostics["targeted_refinement_used"])

    def test_placeholder_foreground_app_gets_independent_browser_identity_audit(self) -> None:
        compact = scene_payload()
        compact["foreground_app_id"] = "current_foreground"
        compact["screen_id"] = "通用动作真机验收页"
        compact["elements"] = [
            {
                "element_id": "page-title",
                "role": "text",
                "meaning": "page_title",
                "label": "通用动作真机验收页",
                "bounds": [100, 100, 700, 180],
                "confidence": 1.0,
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": ["页面顶部唯一主标题"],
            }
        ]
        provider = SequenceProvider(
            [
                compact,
                app_identity_audit_payload(
                    "browser",
                    evidence=["可见浏览器地址栏与页面内容区域"],
                ),
            ]
        )
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={
                "app_id": "current_foreground",
                "objective": "看清当前页面主标题",
            },
        )

        self.assertEqual("browser", scene.foreground_app_id)
        self.assertEqual(2, provider.calls)
        self.assertTrue(
            observer.last_diagnostics["foreground_app_identity_audit_used"]
        )
        audit_messages = provider.messages_seen[1]
        audit_text = audit_messages[1]["content"][0]["text"]
        self.assertNotIn("看清当前页面主标题", audit_text)
        self.assertEqual(
            2,
            len(audit_messages[1]["content"]),
        )

    def test_exact_current_surface_input_skips_unneeded_app_identity_call(self) -> None:
        compact = scene_payload()
        compact["foreground_app_id"] = "current_foreground"
        compact["screen_id"] = "input_page"
        compact["summary"] = "当前唯一正文输入框和中文QWERTY键盘可见"
        compact["elements"] = [
            {
                "element_id": "model-input",
                "role": "input",
                "meaning": "text_input_field",
                "label": "",
                "bounds": [140, 270, 860, 450],
                "confidence": 0.98,
                "states": {
                    "goal_relevant": True,
                    "fully_visible": True,
                    "focused": True,
                    "value": "",
                },
                "evidence": ["正文输入框和光标可见"],
            }
        ]
        input_audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="body",
                    bounds=[140, 270, 860, 450],
                    text="",
                    field_labels=["正文"],
                    caret_line_index=0,
                )
            ],
            keyboard={
                "visible": True,
                "bounds": [80, 570, 920, 1000],
                "layout": "qwerty",
                "input_mode": "chinese_pinyin",
                "case_mode": "lower",
                "qwerty_anchors": {
                    "q": [115, 704],
                    "p": [875, 704],
                    "a": [157, 773],
                    "l": [832, 773],
                    "z": [241, 844],
                    "m": [747, 844],
                    "backspace": [875, 844],
                },
                "mode_switch": None,
            },
        )
        context = {
            "app_id": "current_surface",
            "app_name": "当前界面",
            "objective": "使当前唯一输入框内容精确等于授权文字",
            "entities": {
                "target_surface": "current_surface",
                "input_text": "你好\n世界",
                "target_apps": [],
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_exact_text",
                    "objective": "在当前唯一输入框中逐字输入授权文字",
                    "constraints": ["不要发送或提交"],
                    "completion_conditions": ["输入框逐字等于授权文字"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "target_surface": "current_surface",
                        "input_text": "你好\n世界",
                        "active_input_transaction_text": "你好\n世界",
                        "active_input_field_id": "input_field_1",
                        "active_input_multiline": True,
                    },
                },
            },
        }
        provider = SequenceProvider([compact, input_audit])
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=stable_frames(),
            goal_context=context,
            device_id="device-current-surface-input",
        )

        self.assertEqual(2, provider.calls)
        self.assertEqual("current_foreground", scene.foreground_app_id)
        self.assertFalse(
            observer.last_diagnostics["foreground_app_identity_audit_used"]
        )
        self.assertTrue(observer.last_diagnostics["input_structure_audit_used"])

        named_context = {
            **context,
            "app_id": "chat_app",
            "app_name": "聊天应用",
            "entities": {**context["entities"], "target_apps": ["chat_app"]},
        }
        named_provider = SequenceProvider(
            [compact, app_identity_audit_payload("chat_app"), input_audit]
        )
        named_observer = GenericSceneObserver(named_provider)
        named_scene = named_observer.observe(
            frames=stable_frames((40, 50, 60)),
            goal_context=named_context,
            device_id="device-named-input",
        )

        self.assertEqual(3, named_provider.calls)
        self.assertEqual("chat_app", named_scene.foreground_app_id)
        self.assertTrue(
            named_observer.last_diagnostics[
                "foreground_app_identity_audit_used"
            ]
        )

    def test_unknown_foreground_for_named_app_gets_goal_independent_identity_audit(self) -> None:
        for target_app, audited_app in (
            ("browser", "browser"),
            ("settings", "settings"),
        ):
            with self.subTest(target_app=target_app):
                compact = scene_payload()
                compact["foreground_app_id"] = "unknown"
                compact["screen_id"] = "app_home"
                compact["elements"][0].update(
                    {
                        "meaning": f"open_{target_app}",
                        "label": target_app,
                        "states": {
                            "goal_relevant": True,
                            "fully_visible": True,
                        },
                        "evidence": ["命名目标入口完整可见"],
                    }
                )
                provider = SequenceProvider(
                    [
                        compact,
                        app_identity_audit_payload(audited_app),
                    ]
                )
                observer = GenericSceneObserver(provider)

                scene = observer.observe(
                    frames=stable_frames(),
                    goal_context={
                        "app_id": target_app,
                        "objective": "确认命名目标应用当前在前台",
                    },
                )

                self.assertEqual(audited_app, scene.foreground_app_id)
                self.assertEqual(2, provider.calls)
                audit_text = provider.messages_seen[1][1]["content"][0]["text"]
                self.assertNotIn(target_app, audit_text)
                self.assertTrue(
                    observer.last_diagnostics[
                        "foreground_app_identity_audit_used"
                    ]
                )

    def test_unknown_foreground_without_named_app_does_not_add_identity_call(self) -> None:
        compact = scene_payload()
        compact["foreground_app_id"] = "unknown"
        provider = SequenceProvider([compact])
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={},
        )

        self.assertEqual("unknown", scene.foreground_app_id)
        self.assertEqual(1, provider.calls)
        self.assertFalse(
            observer.last_diagnostics["foreground_app_identity_audit_used"]
        )

    def test_placeholder_foreground_app_audit_is_cross_app_and_low_confidence_fails_closed(self) -> None:
        for audited_app, confidence, expected in (
            ("settings", 0.98, "settings"),
            ("chat_app", 0.98, "chat_app"),
            ("browser", 0.70, "unknown"),
            ("unknown", 1.0, "unknown"),
        ):
            with self.subTest(audited_app=audited_app, confidence=confidence):
                compact = scene_payload()
                compact["foreground_app_id"] = "current_app"
                provider = SequenceProvider(
                    [
                        compact,
                        app_identity_audit_payload(
                            audited_app,
                            confidence=confidence,
                        ),
                    ]
                )
                scene = GenericSceneObserver(provider).observe(
                    frames=stable_frames(),
                    goal_context={},
                )
                self.assertEqual(expected, scene.foreground_app_id)

    def test_existing_structured_foreground_app_does_not_add_identity_call(self) -> None:
        provider = SequenceProvider([scene_payload()])
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={},
        )

        self.assertEqual("calculator", scene.foreground_app_id)
        self.assertEqual(1, provider.calls)
        self.assertFalse(
            observer.last_diagnostics["foreground_app_identity_audit_used"]
        )

    def test_generic_foreground_category_with_named_screen_gets_brand_identity_audit(self) -> None:
        cases = (
            ("messaging", "wechat_main_chat_list", "wechat", "微信"),
            ("system_utility", "settings_main", "settings", "设置"),
        )
        for foreground, screen_id, target_app, app_name in cases:
            with self.subTest(target_app=target_app):
                compact = scene_payload()
                compact["foreground_app_id"] = foreground
                compact["screen_id"] = screen_id
                provider = SequenceProvider(
                    [
                        compact,
                        targeted_delta_payload(elements=compact["elements"]),
                        app_identity_audit_payload(target_app),
                    ]
                )
                observer = GenericSceneObserver(provider)

                scene = observer.observe(
                    frames=stable_frames(),
                    goal_context={
                        "app_id": target_app,
                        "app_name": app_name,
                        "objective": "确认当前命名应用主界面可见",
                    },
                )

                self.assertEqual(target_app, scene.foreground_app_id)
                self.assertEqual(3, provider.calls)
                audit_text = provider.messages_seen[2][1]["content"][0]["text"]
                self.assertNotIn(target_app, audit_text)
                self.assertNotIn(app_name, audit_text)
                self.assertTrue(
                    observer.last_diagnostics[
                        "foreground_app_identity_audit_used"
                    ]
                )

    def test_generic_foreground_category_without_named_page_stays_unrebound(self) -> None:
        compact = scene_payload()
        compact["foreground_app_id"] = "calculator"
        compact["screen_id"] = "main"
        provider = SequenceProvider(
            [compact, targeted_delta_payload(elements=compact["elements"])]
        )
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={"app_id": "settings", "app_name": "设置"},
        )

        self.assertEqual("calculator", scene.foreground_app_id)
        self.assertEqual(2, provider.calls)
        self.assertFalse(
            observer.last_diagnostics["foreground_app_identity_audit_used"]
        )

    def test_runtime_package_foreground_gets_goal_independent_semantic_identity_audit(self) -> None:
        cases = (
            ("com.tencent.mm", "wechat"),
            ("com.android.settings", "settings"),
            ("org.mozilla.firefox", "browser"),
        )
        for runtime_package, semantic_app in cases:
            with self.subTest(runtime_package=runtime_package):
                compact = scene_payload()
                compact["foreground_app_id"] = runtime_package
                provider = SequenceProvider(
                    [
                        compact,
                        targeted_delta_payload(elements=compact["elements"]),
                        app_identity_audit_payload(
                            semantic_app,
                            evidence=["可见应用品牌界面与独立页面结构"],
                        ),
                    ]
                )
                observer = GenericSceneObserver(provider)

                scene = observer.observe(
                    frames=stable_frames(),
                    goal_context={
                        "app_id": semantic_app,
                        "objective": "确认目标页面当前可见",
                    },
                )

                self.assertEqual(semantic_app, scene.foreground_app_id)
                self.assertEqual(3, provider.calls)
                audit_text = provider.messages_seen[2][1]["content"][0]["text"]
                self.assertNotIn(semantic_app, audit_text)
                self.assertTrue(
                    observer.last_diagnostics[
                        "foreground_app_identity_audit_used"
                    ]
                )

    def test_matching_runtime_package_target_does_not_add_identity_call(self) -> None:
        compact = scene_payload()
        compact["foreground_app_id"] = "com.example.reader"
        provider = SequenceProvider(
            [compact, targeted_delta_payload(elements=compact["elements"])]
        )
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={"app_id": "com.example.reader"},
        )

        self.assertEqual("com.example.reader", scene.foreground_app_id)
        self.assertEqual(2, provider.calls)
        self.assertFalse(
            observer.last_diagnostics["foreground_app_identity_audit_used"]
        )

    def test_different_structured_semantic_app_does_not_get_reidentified(self) -> None:
        compact = scene_payload()
        compact["foreground_app_id"] = "calculator"
        provider = SequenceProvider(
            [compact, targeted_delta_payload(elements=compact["elements"])]
        )
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={"app_id": "settings"},
        )

        self.assertEqual("calculator", scene.foreground_app_id)
        self.assertEqual(2, provider.calls)
        self.assertFalse(
            observer.last_diagnostics["foreground_app_identity_audit_used"]
        )

    def test_runtime_package_identity_audit_low_confidence_fails_closed(self) -> None:
        compact = scene_payload()
        compact["foreground_app_id"] = "com.example.reader"
        provider = SequenceProvider(
            [
                compact,
                targeted_delta_payload(elements=compact["elements"]),
                app_identity_audit_payload("reader", confidence=0.70),
            ]
        )

        scene = GenericSceneObserver(provider).observe(
            frames=stable_frames(),
            goal_context={"app_id": "reader"},
        )

        self.assertEqual("unknown", scene.foreground_app_id)
        self.assertEqual(3, provider.calls)

    def test_foreground_app_identity_audit_rejects_unsafe_or_ambiguous_payloads(self) -> None:
        accepted = app_identity_audit_payload(
            "settings",
            evidence=[
                "顶部标题显示设置",
                "搜索系统设置项可见",
                "WLAN与蓝牙菜单可见",
            ],
        )
        app_id, confidence, evidence = _strict_foreground_app_identity_audit(
            json.dumps(accepted, ensure_ascii=False)
        )
        self.assertEqual("settings", app_id)
        self.assertEqual(0.98, confidence)
        self.assertEqual(3, len(evidence))

        invalid_payloads = (
            app_identity_audit_payload("current_foreground"),
            {
                **app_identity_audit_payload("browser"),
                "unexpected": True,
            },
            app_identity_audit_payload(
                "browser",
                evidence=["点击右上角并使用坐标 x=10"],
            ),
            app_identity_audit_payload(
                "browser",
                evidence=["身份线索一", "身份线索二", "身份线索三", "身份线索四"],
            ),
        )
        duplicate = (
            '{"protocol_version":"'
            + FOREGROUND_APP_IDENTITY_AUDIT_VERSION
            + '","foreground_app_id":"browser","foreground_app_id":"settings",'
            '"confidence":0.98,"evidence":["可见应用身份"]}'
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(VisionAgentError):
                    _strict_foreground_app_identity_audit(
                        json.dumps(payload, ensure_ascii=False)
                    )
        with self.assertRaises(VisionAgentError):
            _strict_foreground_app_identity_audit(duplicate)

    def test_foreground_app_identity_audit_filters_unsafe_unknown_evidence(self) -> None:
        payload = app_identity_audit_payload(
            "unknown",
            confidence=0.0,
            evidence=[
                "Title text shows a generic acceptance page",
                "Menu option says Swipe Up",
                "Generic browser navigation bar is visible",
            ],
        )

        app_id, confidence, evidence = _strict_foreground_app_identity_audit(
            json.dumps(payload, ensure_ascii=False)
        )

        self.assertEqual("unknown", app_id)
        self.assertEqual(0.0, confidence)
        self.assertEqual(
            (
                "Title text shows a generic acceptance page",
                "Generic browser navigation bar is visible",
            ),
            evidence,
        )

    def test_foreground_app_identity_prompts_forbid_context_placeholders(self) -> None:
        compact = _compact_prompt(
            {"app_id": "current_foreground", "objective": "读取当前画面"}
        )
        audit = _foreground_app_identity_audit_prompt()

        for prompt in (compact, audit):
            self.assertIn("current_foreground", prompt)
            self.assertIn("unknown", prompt)
        self.assertIn("No user goal", audit)
        self.assertIn("no action authority", audit)

    def test_status_exposes_observation_policy(self) -> None:
        status = GenericSceneObserver(FakeProvider(scene_payload())).status()
        self.assertEqual(status["compact_output_tokens"], 2600)
        self.assertEqual(status["targeted_output_tokens"], 2600)
        self.assertEqual(
            status["targeted_delta_protocol"],
            TARGETED_SCENE_DELTA_PROTOCOL_VERSION,
        )
        self.assertEqual(status["observation_timeout_seconds"], 60.0)
        self.assertEqual(status["max_compact_elements"], 12)
        self.assertEqual(status["current_stage"], "idle")

    def test_ordinal_prompts_require_preceding_visible_siblings(self) -> None:
        context = {"objective": "进入列表中从上往下第二项"}
        compact = _compact_prompt(context)
        targeted = _targeted_prompt(
            context,
            first_scene={
                "foreground_app_id": "unknown",
                "screen_id": "list",
                "summary": "列表页",
                "system_ui": {},
                "overlays": [],
                "confidence": 1.0,
            },
        )

        for prompt in (compact, targeted):
            self.assertIn("之前所有同列", prompt)
            self.assertIn("goal_relevant:true", prompt)
            self.assertIn("不得", prompt)

    def test_title_prompts_prioritize_structured_page_identity(self) -> None:
        context = {"objective": "读取下一页标题"}
        compact = _compact_prompt(context)
        targeted = _targeted_prompt(
            context,
            first_scene={
                "foreground_app_id": "unknown",
                "screen_id": "unknown",
                "summary": "当前页面",
                "system_ui": {},
                "overlays": [],
                "confidence": 1.0,
            },
        )

        for prompt in (compact, targeted):
            self.assertIn("meaning=page_title", prompt)
            self.assertIn("普通正文", prompt)
            self.assertIn("screen_id", prompt)

    def test_clear_goal_recovers_unique_committed_cue_and_visual_row_unit(self) -> None:
        base = _parse_scene(
            json.dumps(scene_payload(), ensure_ascii=False),
            fingerprint="c" * 64,
        )
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    bounds=[140, 570, 720, 650],
                    text="",
                    placeholder="",
                    visible_editable_cues=["first", "|"],
                    caret_line_index=1,
                )
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
        cleared = _apply_input_structure_audit(
            base,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="c" * 64,
            goal_context={
                "objective": "清空当前唯一已聚焦输入框中的全部应用文字，不要发送"
            },
            ledger_input_value=None,
        )
        target = cleared.get_element("local_audited_input_1")
        self.assertEqual("first", target.states["value"])
        self.assertEqual(1, target.states["clear_extra_delete_units"])
        self.assertTrue(target.states["goal_relevant"])
        self.assertTrue(any("保守退格单位" in item for item in target.evidence))

        read_only = _apply_input_structure_audit(
            base,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="d" * 64,
            goal_context={"objective": "读取当前输入框"},
            ledger_input_value=None,
        )
        ordinary = read_only.get_element("local_audited_input_1")
        self.assertEqual("", ordinary.states["value"])
        self.assertNotIn("clear_extra_delete_units", ordinary.states)

    def test_clear_goal_audits_aligned_frames_across_caret_blink_phase(self) -> None:
        compact = scene_payload()
        compact["elements"] = []
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    bounds=[140, 570, 720, 650],
                    text="",
                    placeholder="",
                    visible_editable_cues=["first", "|"],
                    caret_line_index=1,
                )
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
        provider = SequenceProvider([compact, compact, audit])
        scene = GenericSceneObserver(provider).observe(
            frames=stable_frames(),
            goal_context={
                "objective": "清空当前唯一已聚焦输入框中的全部应用文字，不要发送"
            },
        )
        target = scene.get_element("local_audited_input_1")
        self.assertEqual(1, target.states["clear_extra_delete_units"])
        audit_content = provider.messages_seen[-1][1]["content"]
        self.assertEqual(4, len(audit_content))
        self.assertIn("different times", audit_content[0]["text"])
        self.assertIn("caret_line_index", audit_content[0]["text"])

    def test_local_pixels_locate_model_attested_caret_row_from_qwerty_anchor(self) -> None:
        base = _parse_scene(
            json.dumps(scene_payload(), ensure_ascii=False),
            fingerprint="e" * 64,
        )
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    bounds=[140, 570, 720, 650],
                    text="",
                    placeholder="",
                    visible_editable_cues=["first", "vertical text caret"],
                    caret_line_index=None,
                )
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
        frame = Image.new("RGB", (540, 960), (198, 203, 198))
        draw = ImageDraw.Draw(frame)
        draw.rectangle((100, 495, 180, 515), fill=(25, 25, 25))
        draw.rectangle((103, 528, 105, 570), fill=(20, 145, 80))
        frames = tuple(frame.copy() for _ in range(3))
        scene = _apply_input_structure_audit(
            base,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="e" * 64,
            goal_context={
                "objective": "清空当前唯一已聚焦输入框中的全部应用文字，不要发送"
            },
            ledger_input_value=None,
            qwerty_row_snapper=lambda _frames, anchors: anchors,
            qwerty_row_frames=frames,
        )
        target = scene.get_element("local_audited_input_1")
        self.assertEqual(1, target.states["local_caret_line_index"])
        self.assertEqual(1, target.states["clear_extra_delete_units"])
        self.assertTrue(any("本地校准像素" in item for item in target.evidence))

        raw_search_only = _apply_input_structure_audit(
            base,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="a" * 64,
            goal_context={
                "objective": "清空当前唯一已聚焦输入框中的全部应用文字，不要发送"
            },
            ledger_input_value=None,
            qwerty_row_snapper=lambda _frames, _anchors: None,
            qwerty_row_frames=frames,
        )
        raw_target = raw_search_only.get_element("local_audited_input_1")
        self.assertEqual(1, raw_target.states["local_caret_line_index"])
        self.assertEqual(1, raw_target.states["clear_extra_delete_units"])
        self.assertNotIn("qwerty_row_snap", raw_target.evidence)

        audit["application_inputs"][0]["visible_editable_cues"] = ["first"]
        goal_bound_local = _apply_input_structure_audit(
            base,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="f" * 64,
            goal_context={
                "objective": "清空当前唯一已聚焦输入框中的全部应用文字，不要发送"
            },
            ledger_input_value=None,
            qwerty_row_snapper=lambda _frames, anchors: anchors,
            qwerty_row_frames=frames,
        )
        goal_bound_target = goal_bound_local.get_element("local_audited_input_1")
        self.assertEqual(1, goal_bound_target.states["local_caret_line_index"])
        self.assertEqual(1, goal_bound_target.states["clear_extra_delete_units"])

        without_attestation = _apply_input_structure_audit(
            base,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="b" * 64,
            goal_context={"objective": "观察当前输入框，不修改内容"},
            ledger_input_value=None,
            qwerty_row_snapper=lambda _frames, anchors: anchors,
            qwerty_row_frames=frames,
        )
        ordinary = without_attestation.get_element("local_audited_input_1")
        self.assertNotIn("local_caret_line_index", ordinary.states)
        self.assertNotIn("clear_extra_delete_units", ordinary.states)


if __name__ == "__main__":
    unittest.main()
