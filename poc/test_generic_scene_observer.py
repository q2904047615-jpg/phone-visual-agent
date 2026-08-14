from __future__ import annotations

import json
import unittest

from PIL import Image, ImageDraw, ImageFilter

from generic_scene_observer import GenericSceneObserver, INPUT_STRUCTURE_AUDIT_VERSION
from generic_scene_observer import _parse_scene, _scene_enum_values
from ui_scene import UISceneError
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
        "protocol_version": "2026-08-10-ui-scene-v2",
        "foreground_app_id": "calculator",
        "screen_id": "app_home",
        "summary": "计算器首页",
        "system_ui": {
            "immersive_or_fullscreen": False,
            "navigation_bar_visible": True,
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
    return {
        "protocol_version": INPUT_STRUCTURE_AUDIT_VERSION,
        "application_inputs": list(application_inputs or []),
        "ime_preedit_regions": list(ime_preedit_regions or []),
        "keyboard": keyboard
        or {
            "visible": False,
            "bounds": None,
            "layout": "unknown",
            "input_mode": "unknown",
            "mode_switch": None,
        },
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

    def test_goal_relevant_non_input_keyboard_facts_still_fail_closed(self) -> None:
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

        with self.assertRaisesRegex(VisionAgentError, "bad-goal-container"):
            _parse_scene(
                json.dumps(payload, ensure_ascii=False),
                fingerprint="frame-bad-goal-container",
                goal_context={"objective": "切换输入模式"},
            )

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
        self.assertEqual(provider.max_tokens, 800)
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

    def test_invalid_json_gets_one_compact_retry(self) -> None:
        provider = SequenceProvider(["{", scene_payload()])
        observer = GenericSceneObserver(provider)
        scene = observer.observe(frames=stable_frames())
        self.assertEqual(scene.foreground_app_id, "calculator")
        self.assertEqual(provider.calls, 2)
        self.assertEqual(provider.max_tokens_seen, [800, 800])
        self.assertTrue(observer.last_diagnostics["compact_retry_used"])
        self.assertEqual(observer.last_diagnostics["model_calls"], 2)
        self.assertEqual(
            len(observer.last_diagnostics["model_call_elapsed_seconds"]),
            2,
        )
        self.assertGreaterEqual(observer.last_diagnostics["elapsed_seconds"], 0.0)

    def test_bounds_object_retry_prompt_requires_four_number_array(self) -> None:
        invalid = scene_payload()
        invalid["elements"][0]["bounds"] = {
            "x": 100,
            "y": 600,
            "width": 160,
            "height": 160,
        }
        provider = SequenceProvider([invalid, scene_payload()])
        scene = GenericSceneObserver(provider).observe(frames=stable_frames())
        self.assertEqual(scene.elements[0].bounds, (0.1, 0.6, 0.26, 0.76))
        retry_text = provider.messages_seen[1][1]["content"][0]["text"]
        self.assertIn(
            "bounds必须是恰好4个0..1000数值的数组[left,top,right,bottom]",
            retry_text,
        )

    def test_overlay_objects_trigger_one_format_retry_and_keep_candidate_in_elements(self) -> None:
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
        scene = observer.observe(
            frames=stable_frames(),
            goal_context={"objective": "让新的空白页面可见"},
        )

        self.assertEqual(2, provider.calls)
        self.assertEqual("add-new", scene.elements[0].element_id)
        self.assertEqual(("window_manager",), scene.overlays)
        retry_text = provider.messages_seen[1][1]["content"][0]["text"]
        self.assertIn("overlays只能是字符串数组", retry_text)
        self.assertIn("必须改写成elements", retry_text)
        self.assertIn("格式修复不能靠删除真实候选通过", retry_text)
        self.assertIn("即使目标结果", retry_text)
        self.assertIn("尚未出现", retry_text)

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
        self.assertEqual(provider.max_tokens_seen, [800, 1200, 1200])
        self.assertTrue(observer.last_diagnostics["format_retry_used"])
        self.assertTrue(observer.last_diagnostics["repair_retry_success"])

    def test_observation_never_uses_two_format_repairs(self) -> None:
        first_retry = scene_payload()
        first_retry["elements"] = []
        first_retry["summary"] = "未知首页"
        provider = SequenceProvider(["{", first_retry, "{"])
        observer = GenericSceneObserver(provider)
        with self.assertRaises(VisionAgentError):
            observer.observe(
                frames=stable_frames(),
                goal_context={"objective": "查找目标按钮"},
            )
        self.assertEqual(provider.calls, 3)
        self.assertEqual(provider.max_tokens_seen, [800, 800, 1200])
        self.assertTrue(observer.last_diagnostics["format_retry_used"])
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
        self.assertEqual(provider.max_tokens_seen, [800, 1200])
        self.assertEqual(scene.elements[0].label, "微信")
        self.assertTrue(observer.last_diagnostics["targeted_refinement_used"])
        targeted_text = provider.messages_seen[1][1]["content"][0]["text"]
        self.assertIn("置信度只评价当前画面观察本身是否可靠", targeted_text)
        self.assertIn("系统级动作没有屏内按钮", targeted_text)
        self.assertIn("目标相关控件确实不存在时返回空elements", targeted_text)
        self.assertIn("不能因为目标尚未完成而降低", targeted_text)
        self.assertIn("模糊、遮挡或不唯一时仍必须降低", targeted_text)

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
            SequenceProvider([empty, empty, audit])
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
                SequenceProvider([empty, empty, audit])
            ).observe(
                frames=stable_frames(),
                goal_context={"objective": "切换到英文直输模式"},
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
            SequenceProvider([empty, empty, audit])
        ).observe(
            frames=stable_frames(),
            goal_context={"objective": "切换到英文直输模式"},
        )

        self.assertIsNone(scene.unique_trusted_goal_element())
        self.assertFalse(
            any(item.meaning == "switch_keyboard_input_mode" for item in scene.elements)
        )

    def test_targeted_refinement_uses_goal_directed_roi_but_keeps_full_frame_bounds(self) -> None:
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
        targeted_image = provider.messages_seen[1][1]["content"][2]["image_url"]["url"]
        self.assertEqual(compact_image, targeted_overview)
        self.assertNotEqual(compact_image, targeted_image)
        targeted_text = provider.messages_seen[1][1]["content"][0]["text"]
        self.assertIn("局部图只用于看清事实，不增加任何动作权限", targeted_text)
        self.assertIn("所有bounds必须回到第一张完整手机画面", targeted_text)
        self.assertIn("必须在第二张高清局部中重新辨认目标", targeted_text)
        self.assertIn("第二张只提供放大细节，绝不能作为坐标系", targeted_text)

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
        self.assertEqual([800, 1200, 700], provider.max_tokens_seen)

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
        self.assertEqual(status["compact_output_tokens"], 800)
        self.assertEqual(status["observation_timeout_seconds"], 60.0)
        self.assertEqual(status["max_compact_elements"], 12)
        self.assertEqual(status["current_stage"], "idle")


if __name__ == "__main__":
    unittest.main()
