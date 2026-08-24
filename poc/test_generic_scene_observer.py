from __future__ import annotations

import json
import unittest

from PIL import Image, ImageDraw, ImageFilter

from generic_scene_observer import (
    INPUT_STRUCTURE_AUDIT_VERSION,
    SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
    SingleStepGenericSceneObserver,
)
from ui_scene import UI_SCENE_PROTOCOL_VERSION
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


class SingleStepGenericSceneObserverTests(unittest.TestCase):
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

    def test_single_step_non_input_keeps_scene_but_discards_overflow_batch(self) -> None:
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

        self.assertEqual("com.vendor.runtime", observed.foreground_app_id)
        self.assertEqual("sample_app_conversation", observed.screen_id)
        self.assertEqual((), observed.elements)
        self.assertEqual(1, provider.calls)

    def test_single_step_input_keeps_overflow_geometry_fail_closed(self) -> None:
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

        with self.assertRaises(VisionAgentError):
            SingleStepGenericSceneObserver(provider).observe(
                frames=stable_frames(),
                goal_context=context,
                device_id="device-local-01",
            )

        self.assertEqual(1, provider.calls)

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
        observer = SingleStepGenericSceneObserver(provider)
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
