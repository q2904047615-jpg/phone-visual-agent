from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageFilter

from generic_scene_observer import (
    AUDITED_SOFT_KEYBOARD_HIDDEN_EVIDENCE,
    GenericSceneObserver,
    ICON_CLUSTER_AUDIT_VERSION,
    INPUT_STRUCTURE_AUDIT_VERSION,
    SYSTEM_UI_AUDIT_VERSION,
)
from generic_scene_observer import (
    _MAX_JSON_STRUCTURAL_REPAIR_CANDIDATES,
    _MAX_JSON_STRUCTURAL_REPAIR_CHARS,
    _camera_layout_orientation,
    _can_use_stable_ocr_literal_bounds,
    _compact_prompt,
    _goal_requests_input,
    _input_structure_audit_prompt,
    _stable_ocr_literal_bounds,
    _input_structure_diagnostic_shape,
    _parse_scene_after_unique_structural_edit,
    _parse_scene,
    _scene_enum_values,
    _single_json_structural_edits,
    _snap_reload_audit_to_local_glyph,
    _strict_icon_cluster_audit_payload,
    _targeted_prompt,
)
from ocr_runtime import OcrMatch
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
        return json.dumps(self.payload, ensure_ascii=False)


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
) -> dict:
    return {
        "structure_id": structure_id,
        "bounds": bounds or [110, 40, 850, 110],
        "fully_visible": fully_visible,
        "text": text,
        "placeholder": placeholder,
        "visible_editable_cues": ["完整横向输入边框"],
        "confidence": confidence,
        "right_button": right_button,
    }


class GenericSceneObserverTests(unittest.TestCase):
    def test_input_audit_prompt_defines_exact_nullable_mode_switch_contract(self) -> None:
        prompt = _input_structure_audit_prompt(
            {"objective": "切换当前键盘输入模式"},
            roi_bounds=None,
        )

        self.assertIn(INPUT_STRUCTURE_AUDIT_VERSION, prompt)
        self.assertIn('"mode_switch":null', prompt)
        self.assertIn(
            '{"label":"中","bounds":[0,0,1000,1000],"confidence":0.0,'
            '"current_mode":"chinese_pinyin","target_mode":"direct_latin"}',
            prompt,
        )
        self.assertIn(
            '{"label":"英","bounds":[0,0,1000,1000],"confidence":0.0,'
            '"current_mode":"direct_latin","target_mode":"chinese_pinyin"}',
            prompt,
        )
        self.assertIn("Never omit confidence or target_mode", prompt)

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

    def test_observation_prompts_preserve_only_read_only_clipped_list_cue(self) -> None:
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
                        goal_context={"objective": "操作目标控件"},
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
        self.assertTrue(scene.elements[0].states["fully_visible"])
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
                    "external_impact": "navigation_only",
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
                    "external_impact": "navigation_only",
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

    def test_camera_alignment_with_only_forbidden_evidence_remains_fail_closed(self) -> None:
        payload = scene_payload()
        payload["camera_alignment"]["evidence"] = ["顶部PX/MM坐标水平排列"]

        with self.assertRaisesRegex(VisionAgentError, "坐标或控制指令"):
            _parse_scene(
                json.dumps(payload, ensure_ascii=False),
                fingerprint="local-fingerprint",
                goal_context={"objective": "读取当前页面"},
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
        self.assertEqual([1800, 600], provider.max_tokens_seen)
        self.assertTrue(observer.last_diagnostics["system_ui_audit_used"])
        self.assertFalse(observer.last_diagnostics["system_ui_audit_retry_used"])
        self.assertFalse(observer.last_diagnostics["targeted_refinement_used"])
        audit_content = provider.messages_seen[1][1]["content"]
        self.assertEqual(
            3,
            sum(item.get("type") == "image_url" for item in audit_content),
        )

    def test_system_ui_audit_retries_once_then_fails_closed_on_unknown(self) -> None:
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

        self.assertEqual(3, provider.calls)
        self.assertTrue(observer.last_diagnostics["system_ui_audit_used"])
        self.assertTrue(observer.last_diagnostics["system_ui_audit_retry_used"])

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
                self.assertEqual(3, provider.calls)

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

    def test_direction_audit_retries_once_for_rejected_evidence_wording(self):
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

        credential = observer.audit_camera_alignment(
            frames=stable_frames(),
            device_id="device-a",
            scene_fingerprint="scene-a",
        )

        self.assertEqual(2, provider.calls)
        self.assertEqual(("手机页面文字横向排列且字形正立",), credential.evidence)
        diagnostics = observer.last_orientation_audit_diagnostics
        self.assertTrue(diagnostics["audit_accepted"])
        self.assertTrue(diagnostics["retry_used"])
        self.assertEqual(2, diagnostics["model_calls"])
        self.assertEqual(
            [3, 3],
            [
                sum(
                    item.get("type") == "image_url"
                    for item in messages[1]["content"]
                )
                for messages in provider.messages_seen
            ],
        )
        serialized = json.dumps(diagnostics, ensure_ascii=False)
        self.assertNotIn("点击控制端", serialized)
        retry_prompt = provider.messages_seen[1][1]["content"][0]["text"]
        self.assertIn("rejected before any", retry_prompt)
        self.assertIn("physical action", retry_prompt)
        self.assertIn("each at most 60 characters", retry_prompt)

    def test_direction_audit_second_bad_evidence_fails_without_third_call(self):
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

        self.assertEqual(2, provider.calls)
        self.assertEqual(1, len(provider.responses))
        diagnostics = observer.last_orientation_audit_diagnostics
        self.assertFalse(diagnostics["audit_accepted"])
        self.assertTrue(diagnostics["retry_used"])
        self.assertEqual(2, diagnostics["model_calls"])

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
            ({"evidence": ["点击坐标(500,900)"]}, "坐标或控制指令"),
            ({"evidence": ["PX/MM 控制端读数方向正常"]}, "坐标或控制指令"),
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
        self.assertEqual(provider.max_tokens, 1800)
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
        self.assertEqual(provider.call_options["max_attempts"], 2)

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

    def test_low_scene_confidence_accepts_read_only_completion_evidence(self) -> None:
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

    def test_read_only_observation_accepts_one_stale_leading_frame_after_convergence(self) -> None:
        provider = FakeProvider(scene_payload())
        settled = Image.new("RGB", (540, 960), (30, 40, 50))
        stale = Image.new("RGB", settled.size, (255, 255, 255))

        observer = GenericSceneObserver(provider)
        observer.observe(
            frames=[stale, settled.copy(), settled.copy(), settled.copy()]
        )

        self.assertEqual(1, provider.calls)
        self.assertTrue(observer.last_diagnostics["local_stability"]["stable"])

    def test_read_only_observation_never_reselects_ignored_sharp_leading_frame(self) -> None:
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
        self.assertEqual(provider.max_tokens_seen, [1800])
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
        self.assertEqual(provider.max_tokens_seen, [1800])
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
        self.assertEqual(provider.max_tokens_seen, [1800])
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
        self.assertEqual(provider.max_tokens_seen, [1800])
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
        self.assertEqual(provider.max_tokens_seen, [1800])
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

    def test_targeted_invalid_json_uses_the_only_format_retry(self) -> None:
        first = scene_payload()
        first["elements"] = []
        first["summary"] = "未知首页"
        refined = scene_payload()
        provider = SequenceProvider([first, "{", refined])
        observer = GenericSceneObserver(provider)
        scene = observer.observe(
            frames=stable_frames(),
            goal_context={"objective": "查找目标按钮"},
        )
        self.assertEqual(scene.elements[0].element_id, "e1")
        self.assertEqual(provider.calls, 3)
        self.assertEqual(provider.max_tokens_seen, [1800, 1200, 1200])
        self.assertTrue(observer.last_diagnostics["format_retry_used"])
        self.assertTrue(observer.last_diagnostics["repair_retry_success"])

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
        self.assertEqual(provider.max_tokens_seen, [1800])
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
        refined = dict(first)
        refined["elements"] = [
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
        ]
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
        self.assertEqual(provider.max_tokens_seen, [1800, 1200])
        self.assertEqual(scene.elements[0].label, "微信")
        self.assertTrue(observer.last_diagnostics["targeted_refinement_used"])
        targeted_text = provider.messages_seen[1][1]["content"][0]["text"]
        self.assertIn("置信度只评价当前画面观察本身是否可靠", targeted_text)
        self.assertIn("系统级动作没有屏内按钮", targeted_text)
        self.assertIn("目标相关控件确实不存在时返回空elements", targeted_text)
        self.assertIn("不能因为目标尚未完成而降低", targeted_text)
        self.assertIn("模糊、遮挡或不唯一时仍必须降低", targeted_text)

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
        refined = json.loads(json.dumps(first, ensure_ascii=False))
        refined["elements"][0]["element_id"] = "return_entry_01"
        refined["elements"][0]["evidence"] = [
            "说明文字下方带下划线的白色返回入口，四边完整可见"
        ]
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
        refined = dict(first)
        refined["summary"] = "底部边缘存在部分可见的后续内容，列表仍在延伸"
        refined["elements"] = []
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
                goal_context={"objective": "在唯一输入框输入 agent"},
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
        empty = scene_payload()
        empty["elements"] = []
        audit = input_audit_payload(
            application_inputs=[audited_application_input(text=".com")],
            keyboard={
                "visible": True,
                "bounds": [0, 360, 1000, 1000],
                "layout": " Symbols ",
                "input_mode": "chinese_pinyin",
                "mode_switch": None,
            },
        )
        observer = GenericSceneObserver(SequenceProvider([empty, empty, audit]))

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={"objective": "确认输入框内容已经是 .com"},
        )

        candidate = scene.unique_trusted_goal_element()
        self.assertIsNotNone(candidate)
        self.assertEqual("symbol", candidate.states["keyboard_layout"])

    def test_input_audit_keeps_unknown_layout_fail_closed(self) -> None:
        empty = scene_payload()
        empty["elements"] = []
        audit = input_audit_payload(
            keyboard={
                "visible": True,
                "bounds": [0, 360, 1000, 1000],
                "layout": "symbols_custom",
                "input_mode": "unknown",
                "mode_switch": None,
            }
        )

        with self.assertRaisesRegex(VisionAgentError, "keyboard.layout"):
            GenericSceneObserver(SequenceProvider([empty, empty, audit])).observe(
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
            '"input_mode":"unknown","qwerty_anchors":',
            audit_prompt,
        )
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

    def test_keyboard_mode_switch_label_cannot_contradict_current_mode(self) -> None:
        empty = scene_payload()
        empty["elements"] = []
        audit = input_audit_payload(
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
            }
        )

        with self.assertRaisesRegex(VisionAgentError, "label.*current_mode.*冲突"):
            GenericSceneObserver(
                SequenceProvider([empty, empty, audit])
            ).observe(
                frames=stable_frames(),
                goal_context={"objective": "读取当前键盘输入模式"},
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
                    "external_impact": "navigation_only",
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
                    "external_impact": "navigation_only",
                    "goal_entities": {"input_text": "agent"},
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
                    "external_impact": "navigation_only",
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
                    "external_impact": "read_only",
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

    def test_input_goal_keeps_malformed_keyboard_switch_fail_closed(self) -> None:
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
                    "goal_relevant": True,
                    "keyboard_input_mode_switch": True,
                    "current_mode": "direct_latin",
                    "target_mode": "direct_latin",
                },
                "evidence": ["方向与键面文字冲突"],
            }
        ]

        with self.assertRaisesRegex(VisionAgentError, "keyboard_input_mode_switch"):
            GenericSceneObserver(FakeProvider(payload)).observe(
                frames=stable_frames(),
                goal_context={"objective": "让当前输入框显示 agent"},
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
            goal_context={"objective": "顶部右侧圆形箭头对应的页面重新加载已完成"},
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
            goal_context={"objective": "顶部右侧圆形箭头对应的页面重新加载已完成"},
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

    def test_targeted_refinement_uses_only_full_frame_for_actionable_bounds(self) -> None:
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
            goal_context={"objective": "修改顶部已有文字的输入框"},
        )

        self.assertEqual(scene.elements[0].bounds, (0.1, 0.1, 0.9, 0.3))
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
        self.assertEqual([1800, 1200, 700], provider.max_tokens_seen)

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
        self.assertEqual("通用动作真机验收页", scene.screen_id)
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

    def test_status_exposes_observation_policy(self) -> None:
        status = GenericSceneObserver(FakeProvider(scene_payload())).status()
        self.assertEqual(status["compact_output_tokens"], 1800)
        self.assertEqual(status["observation_timeout_seconds"], 60.0)
        self.assertEqual(status["max_compact_elements"], 4)
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


if __name__ == "__main__":
    unittest.main()
