from __future__ import annotations

import ast
import json
import tempfile
import time
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from agent.infrastructure.generic_scene_observer import (
    INPUT_STRUCTURE_AUDIT_VERSION,
    _apply_input_structure_audit,
)
from agent.application.input_value_lineage import (
    describe_input_surface,
    lineage_matches_trailing_newline_cue,
    lineage_matches_visual,
)
from agent.domain.input_value_lineage import (
    InputValueLineageError,
    TYPED_INPUT_LINEAGE_VERSION,
    TypedInputLineage,
    build_pending_chinese_preedit_lineage,
    build_pending_ime_candidate_lineage,
    build_pending_input_state_lineage,
    build_pending_literal_lineage,
    build_pending_newline_lineage,
    build_pending_text_lineage,
)
from agent.infrastructure.file_system_input_lineage_store import (
    FileSystemTypedInputLineageStore as TypedInputLineageStore,
)
from agent.domain.ui_scene import UIScene
from agent.domain.vision_model import VisionAgentError


DEVICE = "device-test-01"
PRIOR = "long2026:123@"
EXPECTED = PRIOR + "7"
RAW_AFTER = "long2026:\n123@7"


def surface_frame(*, unrelated: bool = False, variation: int = 0) -> Image.Image:
    image = Image.new("RGB", (810, 1440), "white" if not unrelated else "#202020")
    draw = ImageDraw.Draw(image)
    if unrelated:
        draw.rectangle((100, 770, 710, 930), fill="#101010", outline="#f00000", width=8)
        draw.line((100, 930, 710, 770), fill="#ffffff", width=12)
    else:
        draw.rounded_rectangle(
            (105, 775, 560, 910),
            radius=20,
            fill="#f4f4f4",
            outline="#b0b0b0",
            width=3,
        )
        draw.line((135, 825 + variation, 520, 825 + variation), fill="#303030", width=5)
        draw.line((135, 862 + variation, 420, 862 + variation), fill="#303030", width=5)
    return image


def surface_frames() -> tuple[Image.Image, ...]:
    return tuple(surface_frame(variation=index % 2) for index in range(4))


def scene(
    value: str,
    fingerprint: str,
    *,
    bounds=(0.13, 0.54, 0.69, 0.61),
    app_id="sample.app",
    screen_id="editor",
) -> dict:
    return {
        "protocol_version": "2026-08-14-ui-scene-v3",
        "foreground_app_id": app_id,
        "app_id": app_id,
        "screen_id": screen_id,
        "summary": "唯一输入框和键盘可见",
        "system_ui": {
            "immersive_or_fullscreen": False,
            "navigation_bar_visible": True,
        },
        "camera_alignment": {
            "camera_layout_orientation": "portrait",
            "phone_content_rotation": "upright",
            "confidence": 1.0,
            "evidence": ["页面文字水平排列"],
        },
        "elements": [
            {
                "element_id": "input-1",
                "role": "input",
                "meaning": "application_text_input",
                "bounds": list(bounds),
                "confidence": 1.0,
                "label": value,
                "states": {
                    "goal_relevant": False,
                    "fully_visible": True,
                    "focused": True,
                    "value": value,
                },
                "evidence": [f"应用输入框当前文字：{value}"],
            }
        ],
        "overlays": ["软键盘可见"],
        "stable": True,
        "confidence": 1.0,
        "fingerprint": fingerprint,
    }


def resolved() -> dict:
    return {
        "node_id": "node-1",
        "kind": "tap_semantic",
        "normalized_point": [0.5, 0.8],
        "normalized_end_point": None,
        "text": None,
        "input_fragment": None,
        "input_method": None,
        "input_pinyin": None,
        "prior_input_value": PRIOR,
        "expected_input_value": EXPECTED,
        "delete_count": None,
        "direction": None,
        "hold_seconds": None,
        "path_distance": None,
        "target_element_id": "key-7",
        "destination_element_id": None,
        "before_fingerprint": "before-fp",
        "expected_effect": {
            "element_state": {
                "meaning": "application_text_input",
                "states": {"value": EXPECTED},
            }
        },
        "formal_candidate_id": "candidate-1",
        "formal_transition": {},
    }


def before_scene() -> dict:
    value = scene(PRIOR, "before-fp")
    value["elements"].append(
        {
            "element_id": "key-7",
            "role": "button",
            "meaning": "input_exact_literal_key",
            "bounds": [0.2, 0.75, 0.3, 0.82],
            "confidence": 1.0,
            "label": "7",
            "states": {
                "goal_relevant": True,
                "fully_visible": True,
                "input_literal_key": True,
                "key_value": "7",
                "prior_input_value": PRIOR,
                "expected_input_value": EXPECTED,
                "input_element_id": "input-1",
                "independent_geometry_verified": True,
                "geometry_audit_source": "input_structure_audit",
            },
            "evidence": ["唯一完整可见键位"],
        }
    )
    return value


def receipt() -> dict:
    return {
        "version": "test-click-barrier-v1",
        "seller_event_barrier_confirmed": True,
        "round_trip_position_confirmed": True,
        "mechanical_contact_ack": False,
    }


def newline_case(prior: str = "first") -> tuple[dict, dict, dict]:
    expected = prior + "\n"
    before = scene(prior, "newline-before")
    before_input = before["elements"][0]
    before_input["states"].update(
        {"input_field_id": "input_field_1", "input_multiline": True}
    )
    before["elements"].append(
        {
            "element_id": "enter-1",
            "role": "button",
            "meaning": "input_exact_enter_key",
            "bounds": [0.82, 0.89, 0.94, 0.96],
            "confidence": 1.0,
            "label": "↵",
            "states": {
                "goal_relevant": True,
                "fully_visible": True,
                "input_enter_key": True,
                "key_action": "newline",
                "key_value": "\n",
                "prior_input_value": prior,
                "expected_input_value": expected,
                "input_element_id": "input-1",
                "input_field_id": "input_field_1",
            },
            "evidence": ["唯一完整可见换行键"],
        }
    )
    action = {
        **resolved(),
        "kind": "press_enter",
        "prior_input_value": prior,
        "expected_input_value": expected,
        "target_element_id": "enter-1",
        "before_fingerprint": "newline-before",
        "expected_effect": {
            "element_state": {
                "meaning": "application_text_input",
                "states": {"value": expected},
            }
        },
    }
    after = scene(expected, "newline-after")
    after_input = after["elements"][0]
    after_input["states"].update(
        {
            "input_field_id": "input_field_1",
            "input_multiline": True,
            "verified_trailing_newline": True,
        }
    )
    after_input["evidence"].append(
        "已验证换行动作、同一typed输入框、精确可见前缀与下一行光标一致"
    )
    return action, before, after


def state_switch_case(
    meaning: str = "switch_keyboard_layout",
) -> tuple[dict, dict]:
    state_key, current_key, target_key, current, target = {
        "switch_keyboard_layout": (
            "keyboard_layout",
            "current_layout",
            "target_layout",
            "qwerty",
            "numeric",
        ),
        "switch_keyboard_case": (
            "keyboard_case_mode",
            "current_mode",
            "target_mode",
            "lower",
            "upper",
        ),
        "switch_keyboard_input_mode": (
            "keyboard_input_mode",
            "current_mode",
            "target_mode",
            "direct_latin",
            "chinese_pinyin",
        ),
    }[meaning]
    before = scene(PRIOR, "before-state-fp")
    before["elements"][0]["states"].update(
        {
            "keyboard_layout": "qwerty",
            "keyboard_input_mode": "direct_latin",
            "keyboard_case_mode": "lower",
        }
    )
    before["elements"].append(
        {
            "element_id": "state-switch-1",
            "role": "button",
            "meaning": meaning,
            "bounds": [0.1, 0.86, 0.25, 0.94],
            "confidence": 1.0,
            "label": target,
            "states": {
                "goal_relevant": True,
                "fully_visible": True,
                current_key: current,
                target_key: target,
                "prior_input_value": PRIOR,
                "input_element_id": "input-1",
            },
            "evidence": ["唯一完整可见输入状态切换键"],
        }
    )
    resolved_state = {
        **resolved(),
        "node_id": "state-switch-node",
        "prior_input_value": PRIOR,
        "expected_input_value": PRIOR,
        "target_element_id": "state-switch-1",
        "before_fingerprint": "before-state-fp",
        "expected_effect": {
            "element_state": {
                "meaning": "application_text_input",
                "states": {"value": PRIOR, state_key: target},
            }
        },
    }
    return before, resolved_state


def ime_candidate_case(
    *,
    prior: str = "",
    segment: str = "loopok",
    pinyin: str | None = None,
    app_id: str = "sample.app",
    screen_id: str = "editor",
    bounds=(0.13, 0.54, 0.69, 0.61),
    input_mode: str = "chinese_pinyin",
) -> tuple[dict, dict]:
    expected = prior + segment
    preedit = pinyin or segment
    before = scene(
        prior,
        "before-ime-candidate-fp",
        bounds=bounds,
        app_id=app_id,
        screen_id=screen_id,
    )
    before_input = before["elements"][0]
    before_input["states"].update(
        {
            "input_field_id": "input_field_1",
            "input_multiline": "\n" in expected,
            "keyboard_layout": "qwerty",
            "keyboard_input_mode": input_mode,
            "keyboard_case_mode": "lower",
            "ime_preedit_text": preedit,
            "ime_exact_candidate_text": segment,
        }
    )
    before["elements"].append(
        {
            "element_id": "candidate-loopok",
            "role": "button",
            "meaning": "ime_exact_candidate",
            "bounds": [0.08, 0.62, 0.22, 0.65],
            "confidence": 1.0,
            "label": segment,
            "states": {
                "goal_relevant": True,
                "fully_visible": True,
                "ime_candidate": True,
                "input_element_id": "input-1",
                "prior_input_value": prior,
                "expected_input_value": expected,
                "pinyin": preedit,
                "independent_geometry_verified": True,
                "geometry_audit_source": "input_structure_audit",
            },
            "evidence": ["唯一完整精确候选"],
        }
    )
    action = {
        **resolved(),
        "node_id": "candidate-node",
        "prior_input_value": prior,
        "expected_input_value": expected,
        "target_element_id": "candidate-loopok",
        "before_fingerprint": "before-ime-candidate-fp",
        "expected_effect": {
            "element_state": {
                "meaning": "application_text_input",
                "states": {"value": expected},
            }
        },
    }
    return before, action


def resolved_text(*, prior: str = "", fragment: str = "longinput") -> dict:
    expected = prior + fragment
    return {
        "node_id": "node-text-1",
        "kind": "input_verified_text",
        "normalized_point": [0.4, 0.6],
        "normalized_end_point": None,
        "text": expected,
        "input_fragment": fragment,
        "input_method": "direct_latin",
        "input_pinyin": None,
        "prior_input_value": prior,
        "expected_input_value": expected,
        "delete_count": None,
        "direction": None,
        "hold_seconds": None,
        "path_distance": None,
        "target_element_id": "input-1",
        "destination_element_id": None,
        "before_fingerprint": "before-fp",
        "expected_effect": {
            "element_state": {
                "meaning": "application_text_input",
                "states": {"value": expected},
            }
        },
        "formal_candidate_id": "candidate-text-1",
        "formal_transition": {},
    }


def resolved_chinese_preedit(
    *,
    prior: str = "",
    fragment: str = "你好",
    pinyin: str = "nihao",
    target_text: str | None = None,
) -> dict:
    expected = prior + fragment
    action = resolved_text(prior=prior, fragment=fragment)
    action.update(
        {
            "text": target_text if target_text is not None else expected,
            "input_method": "chinese_pinyin",
            "input_pinyin": pinyin,
            "expected_effect": {
                "element_state": {
                    "meaning": "application_text_input",
                    "states": {
                        "value": prior,
                        "ime_preedit_text": pinyin,
                        "ime_exact_candidate_text": fragment,
                    },
                }
            },
        }
    )
    return action


def input_audit_raw(value: str, literal: str | None = "x") -> str:
    return json.dumps(
        {
            "protocol_version": INPUT_STRUCTURE_AUDIT_VERSION,
            "application_inputs": [
                {
                    "structure_id": "input-1",
                    "bounds": [130, 540, 690, 610],
                    "fully_visible": True,
                    "text": value,
                    "placeholder": "",
                    "visible_editable_cues": ["cursor"],
                    "caret_line_index": None,
                    "confidence": 1.0,
                    "right_button": None,
                }
            ],
            "ime_preedit_regions": [],
            "keyboard": {
                "visible": True,
                "bounds": [0, 700, 1000, 1000],
                "layout": "symbol",
                "input_mode": "direct_latin",
                "mode_switch": None,
                "literal_keys": ([] if literal is None else [
                    {
                        "label": literal,
                        "value": literal,
                        "bounds": [200, 760, 300, 830],
                        "confidence": 1.0,
                        "fully_visible": True,
                        "key_kind": "character",
                    }
                ]),
            },
        },
        ensure_ascii=False,
    )


def state_switch_audit_raw(*, cue: str = PRIOR, literal: str = "2") -> str:
    """Replay the generic shape returned after a keyboard layout switch."""

    return json.dumps(
        {
            "protocol_version": INPUT_STRUCTURE_AUDIT_VERSION,
            "application_inputs": [
                {
                    "structure_id": "app-input-1",
                    "bounds": [130, 540, 690, 600],
                    "fully_visible": True,
                    "text": "",
                    "placeholder": "",
                    "visible_editable_cues": ([] if not cue else [cue]),
                    "caret_line_index": None,
                    "confidence": 1.0,
                    "right_button": {
                        "label": "发送",
                        "bounds": [780, 540, 920, 600],
                    },
                }
            ],
            "ime_preedit_regions": [
                {
                    "region_id": "ime-preedit-1",
                    "bounds": [100, 600, 900, 660],
                    "text": cue,
                    "confidence": 1.0,
                    "candidates": [],
                }
            ],
            "keyboard": {
                "visible": True,
                "bounds": [0, 660, 1000, 1000],
                "layout": "numeric",
                "input_mode": "direct_latin",
                "case_mode": "unknown",
                "qwerty_anchors": None,
                "mode_switch": None,
                "backspace_key": {
                    "label": "",
                    "bounds": [820, 670, 980, 730],
                    "confidence": 1.0,
                    "fully_visible": True,
                },
                "case_switch": None,
                "literal_keys": [
                    {
                        "value": literal,
                        "label": literal,
                        "key_kind": "character",
                        "bounds": [420, 670, 580, 730],
                        "confidence": 1.0,
                        "fully_visible": True,
                    }
                ],
                "layout_switches": [
                    {
                        "label": "!?#",
                        "bounds": [20, 890, 180, 950],
                        "confidence": 1.0,
                        "current_layout": "numeric",
                        "target_layout": "symbol",
                    }
                ],
            },
        },
        ensure_ascii=False,
    )


def keyboard_routing_audit_raw(
    *,
    value: str,
    layout: str,
    input_mode: str,
    literal: str | None = None,
    mode_switch: dict | None = None,
    layout_switches: list[dict] | None = None,
) -> str:
    qwerty_anchors = (
        {
            "q": [122, 710],
            "p": [880, 710],
            "a": [164, 782],
            "l": [838, 782],
            "z": [248, 853],
            "m": [754, 853],
            "backspace": [880, 853],
        }
        if layout == "qwerty"
        else None
    )
    return json.dumps(
        {
            "protocol_version": INPUT_STRUCTURE_AUDIT_VERSION,
            "application_inputs": [
                {
                    "structure_id": "app-input-1",
                    "bounds": [130, 540, 690, 610],
                    "fully_visible": True,
                    "text": value,
                    "placeholder": "",
                    "visible_editable_cues": ["cursor"],
                    "caret_line_index": None,
                    "confidence": 1.0,
                    "right_button": None,
                }
            ],
            "ime_preedit_regions": [],
            "keyboard": {
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
                "literal_keys": (
                    []
                    if literal is None
                    else [
                        {
                            "value": literal,
                            "label": literal,
                            "key_kind": "character",
                            "bounds": [420, 710, 580, 780],
                            "confidence": 1.0,
                            "fully_visible": True,
                        }
                    ]
                ),
                "layout_switches": list(layout_switches or []),
            },
        },
        ensure_ascii=False,
    )


def ime_commit_audit_raw(*, cue: str = "loopok") -> str:
    """Replay an exact candidate committed after the placeholder disappeared."""

    return json.dumps(
        {
            "protocol_version": INPUT_STRUCTURE_AUDIT_VERSION,
            "application_inputs": [
                {
                    "structure_id": "app-input-1",
                    "bounds": [140, 540, 700, 610],
                    "fully_visible": True,
                    "text": "",
                    "placeholder": "",
                    "visible_editable_cues": ([] if not cue else [cue]),
                    "caret_line_index": None,
                    "confidence": 1.0,
                    "right_button": None,
                }
            ],
            "ime_preedit_regions": [],
            "keyboard": {
                "visible": True,
                "bounds": [0, 660, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "case_mode": "lower",
                "qwerty_anchors": {
                    "q": [122, 710],
                    "p": [880, 710],
                    "a": [164, 782],
                    "l": [838, 782],
                    "z": [248, 853],
                    "m": [754, 853],
                    "backspace": [880, 853],
                },
                "mode_switch": None,
                "backspace_key": None,
                "enter_key": None,
                "case_switch": None,
                "literal_keys": [],
                "layout_switches": [],
            },
        },
        ensure_ascii=False,
    )


def ime_prediction_commit_audit_raw(
    *,
    exact_value: str,
    candidate_text: str,
    caret_marker: str = "|",
    preedit_text: str | None = None,
    application_text: str | None = None,
    visible_editable_cues: list[str] | None = None,
    input_mode: str = "chinese_pinyin",
) -> str:
    """Replay a committed candidate while the IME prediction row remains."""

    committed_segment = preedit_text or candidate_text
    return json.dumps(
        {
            "protocol_version": INPUT_STRUCTURE_AUDIT_VERSION,
            "application_inputs": [
                {
                    "structure_id": "app-input-1",
                    "bounds": [140, 270, 860, 450],
                    "fully_visible": True,
                    "text": application_text or "",
                    "placeholder": "",
                    "field_labels": ["正文"],
                    "visible_editable_cues": (
                        list(visible_editable_cues)
                        if visible_editable_cues is not None
                        else (
                            ["caret"]
                            if application_text is not None
                            else [exact_value + caret_marker]
                        )
                    ),
                    "caret_line_index": exact_value.count("\n"),
                    "confidence": 1.0,
                    "right_button": None,
                }
            ],
            "ime_preedit_regions": [
                {
                    "region_id": "ime-preedit-1",
                    "bounds": [140, 270, 860, 450],
                    "text": committed_segment,
                    "confidence": 1.0,
                    "candidates": [
                        {
                            "text": committed_segment,
                            "bounds": [110, 590, 230, 630],
                            "confidence": 1.0,
                            "fully_visible": True,
                        },
                        {
                            "text": "候选",
                            "bounds": [260, 590, 360, 630],
                            "confidence": 1.0,
                            "fully_visible": True,
                        },
                    ],
                }
            ],
            "keyboard": {
                "visible": True,
                "bounds": [80, 570, 920, 1000],
                "layout": "qwerty",
                "input_mode": input_mode,
                "case_mode": "lower",
                "qwerty_anchors": {
                    "q": [120, 710],
                    "p": [860, 710],
                    "a": [160, 790],
                    "l": [800, 790],
                    "z": [260, 870],
                    "m": [720, 870],
                    "backspace": [840, 870],
                },
                "mode_switch": None,
                "backspace_key": None,
                "enter_key": None,
                "case_switch": None,
                "literal_keys": [],
                "layout_switches": [],
            },
        },
        ensure_ascii=False,
    )


class TypedInputLineageTests(unittest.TestCase):
    def make_store(self, root: str, now: float = 1000.0) -> TypedInputLineageStore:
        return TypedInputLineageStore(Path(root), clock=lambda: now)

    def test_input_lineage_builders_share_one_record_authority(self) -> None:
        source = (
            Path(__file__).resolve().parent
            / "agent"
            / "domain"
            / "input_value_lineage.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        builder_names = (
            "build_pending_text_lineage",
            "build_pending_chinese_preedit_lineage",
            "build_pending_newline_lineage",
            "build_pending_input_state_lineage",
            "build_pending_ime_candidate_lineage",
            "build_pending_literal_lineage",
            "build_verified_text_lineage",
            "build_verified_newline_lineage",
            "build_verified_literal_lineage",
        )

        self.assertEqual(1, source.count("def _build_input_lineage("))
        for resolver in (
            "_resolve_pending_input_surface",
            "_resolve_verified_input_surface",
        ):
            self.assertEqual(1, source.count(f"def {resolver}("))
        for name in builder_names:
            resolver = (
                "_resolve_pending_input_surface"
                if name.startswith("build_pending_")
                else "_resolve_verified_input_surface"
            )
            direct_calls = [
                call
                for call in ast.walk(functions[name])
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            ]
            call_names = [call.func.id for call in direct_calls]
            with self.subTest(builder=name):
                self.assertIn(resolver, call_names)
                self.assertEqual(1, call_names.count("_build_input_lineage"))
                self.assertNotIn("TypedInputLineage", call_names)

    def test_compact_input_value_shadow_authority_is_physically_absent(self) -> None:
        observer_source = (
            Path(__file__).resolve().parent
            / "agent"
            / "infrastructure"
            / "generic_scene_observer.py"
        ).read_text(encoding="utf-8")
        lineage_source = (
            Path(__file__).resolve().parent
            / "agent"
            / "domain"
            / "input_value_lineage.py"
        ).read_text(encoding="utf-8")
        for retired in (
            "def _unique_scene_input_value(",
            "def _unique_payload_input_value(",
            "def _single_step_preliminary_input_attestation(",
            "preliminary_input_value_hint",
            "single_step_input_attestation",
            "single_step_empty_field_evidence",
            "same_frame_visible_cue_text",
            "visible_trailing_newline_projection",
            "coarse_input_value",
        ):
            with self.subTest(retired=retired):
                self.assertNotIn(retired, observer_source)
        self.assertNotIn("coarse_exact_value", lineage_source)
        self.assertIn(
            "def _resolve_pending_ime_candidate_input_state(",
            observer_source,
        )




    def test_unknown_app_requires_related_screen_and_distinctive_value(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = self.make_store(temp)
            record = store.record_verified_literal_action(
                device_id=DEVICE,
                resolved_action=resolved(),
                before_scene=before_scene(),
                after_scene=scene(RAW_AFTER, "after-fp"),
                hardware_receipt=receipt(),
                after_frames=surface_frames(),
            )
            self.assertTrue(
                lineage_matches_visual(
                    record,
                    device_id=DEVICE,
                    app_id="unknown",
                    screen_id="editor_composing",
                    raw_value=RAW_AFTER,
                    now_epoch=1000.0,
                )
            )
            self.assertFalse(
                lineage_matches_visual(
                    record,
                    device_id=DEVICE,
                    app_id="another.app",
                    screen_id="editor_composing",
                    raw_value=RAW_AFTER,
                    now_epoch=1000.0,
                )
            )

    def test_local_surface_descriptor_bridges_arbitrary_model_identity_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = self.make_store(temp)
            record = store.record_verified_literal_action(
                device_id=DEVICE,
                resolved_action=resolved(),
                before_scene=before_scene(),
                after_scene=scene(RAW_AFTER, "after-fp"),
                hardware_receipt=receipt(),
                after_frames=surface_frames(),
            )
            self.assertTrue(
                lineage_matches_visual(
                    record,
                    device_id=DEVICE,
                    app_id="arbitrary.model.name",
                    screen_id="unrelated_model_screen_name",
                    raw_value=RAW_AFTER,
                    input_bounds=(0.13, 0.59, 0.68, 0.69),
                    now_epoch=1000.0,
                    current_frame=surface_frame(variation=1),
                )
            )
            self.assertFalse(
                lineage_matches_visual(
                    record,
                    device_id=DEVICE,
                    app_id="arbitrary.model.name",
                    screen_id="unrelated_model_screen_name",
                    raw_value=RAW_AFTER,
                    input_bounds=(0.13, 0.59, 0.68, 0.69),
                    now_epoch=1000.0,
                    current_frame=surface_frame(unrelated=True),
                )
            )
            self.assertFalse(
                lineage_matches_visual(
                    record,
                    device_id=DEVICE,
                    app_id="arbitrary.model.name",
                    screen_id="unrelated_model_screen_name",
                    raw_value=RAW_AFTER,
                    input_bounds=(0.13, 0.59, 0.68, 0.69),
                    now_epoch=1000.0,
                )
            )
            self.assertFalse(
                lineage_matches_visual(
                    record,
                    device_id=DEVICE,
                    app_id="unknown",
                    screen_id="unrelated_surface",
                    raw_value=RAW_AFTER,
                    now_epoch=1000.0,
                )
            )


    def test_persisted_surface_exact_adjacent_cue_recovers_value(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = TypedInputLineageStore(Path(temp))
            record = store.record_verified_literal_action(
                device_id=DEVICE,
                resolved_action=resolved(),
                before_scene=before_scene(),
                after_scene=scene(RAW_AFTER, "after-fp"),
                hardware_receipt=receipt(),
                after_frames=surface_frames(),
            )
            base = UIScene.from_dict(scene("", "current-fp"))
            audited = _apply_input_structure_audit(
                base,
                state_switch_audit_raw(cue=EXPECTED, literal="2"),
                fingerprint="current-fp",
                goal_context={
                    "objective": f"让输入框逐字显示 {EXPECTED}2",
                    "entities": {"input_text": EXPECTED + "2"},
                },
                ledger_input_value=EXPECTED,
                verified_input_lineage=record,
                device_id=DEVICE,
                lineage_frame=surface_frame(variation=1),
            )
            input_element = audited.get_element("local_audited_input_1")
            self.assertEqual(EXPECTED, input_element.states["value"])
            self.assertTrue(
                any("持久回执连续性" in item for item in input_element.evidence)
            )

    def test_next_verified_key_inherits_prior_known_surface_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = self.make_store(temp)
            first = store.record_verified_literal_action(
                device_id=DEVICE,
                resolved_action=resolved(),
                before_scene=before_scene(),
                after_scene=scene(RAW_AFTER, "after-fp"),
                hardware_receipt=receipt(),
                after_frames=surface_frames(),
            )
            next_prior = EXPECTED
            next_expected = next_prior + "."
            next_action = resolved()
            next_action["prior_input_value"] = next_prior
            next_action["expected_input_value"] = next_expected
            next_action["target_element_id"] = "key-dot"
            next_action["expected_effect"]["element_state"]["states"] = {
                "value": next_expected
            }
            next_before = scene(
                next_prior,
                "after-fp",
                app_id="unknown",
                screen_id="editor_composing",
            )
            next_before["elements"].append(
                {
                    "element_id": "key-dot",
                    "role": "button",
                    "meaning": "input_exact_literal_key",
                    "bounds": [0.2, 0.75, 0.3, 0.82],
                    "confidence": 1.0,
                    "label": ".",
                    "states": {
                        "goal_relevant": True,
                        "fully_visible": True,
                        "input_literal_key": True,
                        "key_value": ".",
                        "prior_input_value": next_prior,
                        "expected_input_value": next_expected,
                        "input_element_id": "input-1",
                        "independent_geometry_verified": True,
                        "geometry_audit_source": "input_structure_audit",
                    },
                    "evidence": ["唯一完整可见键位"],
                }
            )
            second = store.record_verified_literal_action(
                device_id=DEVICE,
                resolved_action=next_action,
                before_scene=next_before,
                after_scene=scene(
                    "long2026:\n123@7.",
                    "next-fp",
                    app_id="unknown",
                    screen_id="editor_composing",
                ),
                hardware_receipt=receipt(),
                after_frames=surface_frames(),
            )
            self.assertEqual(second.exact_value, next_expected)
            self.assertEqual(second.app_id, first.app_id)
            self.assertEqual(second.screen_id, first.screen_id)

    def test_invalid_receipt_or_exact_chain_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = self.make_store(temp)
            bad_receipt = receipt()
            bad_receipt["seller_event_barrier_confirmed"] = False
            with self.assertRaises(InputValueLineageError):
                store.record_verified_literal_action(
                    device_id=DEVICE,
                    resolved_action=resolved(),
                    before_scene=before_scene(),
                    after_scene=scene(RAW_AFTER, "after-fp"),
                    hardware_receipt=bad_receipt,
                    after_frames=surface_frames(),
                )
            bad_action = resolved()
            bad_action["expected_input_value"] = EXPECTED + "x"
            with self.assertRaises(InputValueLineageError):
                store.record_verified_literal_action(
                    device_id=DEVICE,
                    resolved_action=bad_action,
                    before_scene=before_scene(),
                    after_scene=scene(RAW_AFTER, "after-fp"),
                    hardware_receipt=receipt(),
                    after_frames=surface_frames(),
                )

    def test_pending_lineage_uses_same_exact_chain_without_persisting(self) -> None:
        record = build_pending_literal_lineage(
            device_id=DEVICE,
            resolved_action=resolved(),
            before_scene=before_scene(),
            hardware_receipt=receipt(),
        )
        self.assertEqual(record.exact_value, EXPECTED)
        self.assertEqual(record.source, "pending_verified_literal_action")
        self.assertTrue(
            lineage_matches_visual(
                record,
                device_id=DEVICE,
                app_id="sample.app",
                screen_id="editor",
                raw_value=RAW_AFTER,
                input_bounds=(0.13, 0.54, 0.69, 0.61),
            )
        )

    def test_pending_input_state_lineage_accepts_only_three_typed_switches(self) -> None:
        for meaning in (
            "switch_keyboard_layout",
            "switch_keyboard_case",
            "switch_keyboard_input_mode",
        ):
            with self.subTest(meaning=meaning):
                before, action = state_switch_case(meaning)
                record = build_pending_input_state_lineage(
                    device_id=DEVICE,
                    resolved_action=action,
                    before_scene=before,
                    hardware_receipt=receipt(),
                    recorded_at_epoch=1000.0,
                )
                self.assertEqual("pending_verified_input_state_action", record.source)
                self.assertEqual(PRIOR, record.exact_value)
                self.assertTrue(
                    record.matches_pending_input_state_surface(
                        device_id=DEVICE,
                        app_id="sample.app",
                        screen_id="editor",
                        input_bounds=(0.13, 0.54, 0.69, 0.61),
                        now_epoch=1000.0,
                    )
                )
                self.assertTrue(
                    record.matches_pending_input_state_value(
                        device_id=DEVICE,
                        app_id="sample.app",
                        screen_id="editor",
                        raw_value=PRIOR,
                        input_bounds=(0.13, 0.54, 0.69, 0.61),
                        now_epoch=1000.0,
                    )
                )

    def test_pending_typed_auxiliary_actions_accept_unknown_app(self) -> None:
        literal_before = before_scene()
        literal_before["foreground_app_id"] = "unknown"
        literal_before["app_id"] = "unknown"
        literal_before["elements"][0]["states"][
            "input_field_id"
        ] = "input_field_1"
        literal = build_pending_literal_lineage(
            device_id=DEVICE,
            resolved_action=resolved(),
            before_scene=literal_before,
            hardware_receipt=receipt(),
            recorded_at_epoch=1000.0,
        )
        self.assertEqual("input_field_1", literal.input_field_id)
        self.assertTrue(
            literal.matches_typed_context(
                device_id=DEVICE,
                app_id="unknown",
                screen_id="editor",
                input_field_id="input_field_1",
                now_epoch=1000.0,
            )
        )

        state_before, state_action = state_switch_case()
        state_before["foreground_app_id"] = "unknown"
        state_before["app_id"] = "unknown"
        state_before["elements"][0]["states"][
            "input_field_id"
        ] = "input_field_1"
        state = build_pending_input_state_lineage(
            device_id=DEVICE,
            resolved_action=state_action,
            before_scene=state_before,
            hardware_receipt=receipt(),
            recorded_at_epoch=1000.0,
        )
        self.assertEqual("input_field_1", state.input_field_id)
        self.assertTrue(
            state.matches_typed_context(
                device_id=DEVICE,
                app_id="unknown",
                screen_id="editor",
                input_field_id="input_field_1",
                now_epoch=1000.0,
            )
        )

    def test_verified_text_and_literal_require_same_typed_unknown_app_field(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as text_temp:
            text_before = scene(
                "",
                "before-fp",
                app_id="unknown",
            )
            text_after = scene(
                "longinput",
                "after-fp",
                app_id="unknown",
            )
            for candidate in (text_before, text_after):
                candidate["elements"][0]["states"][
                    "input_field_id"
                ] = "input_field_1"
            text_store = TypedInputLineageStore(Path(text_temp))
            text_record = text_store.record_verified_text_action(
                device_id=DEVICE,
                resolved_action=resolved_text(),
                before_scene=text_before,
                after_scene=text_after,
                after_frames=surface_frames(),
            )
            self.assertEqual("unknown", text_record.app_id)
            self.assertEqual("input_field_1", text_record.input_field_id)

        with tempfile.TemporaryDirectory() as literal_temp:
            literal_before = before_scene()
            literal_after = scene(
                RAW_AFTER,
                "after-fp",
                app_id="unknown",
            )
            literal_before["foreground_app_id"] = "unknown"
            literal_before["app_id"] = "unknown"
            for candidate in (literal_before, literal_after):
                candidate["elements"][0]["states"][
                    "input_field_id"
                ] = "input_field_1"
            literal_store = TypedInputLineageStore(Path(literal_temp))
            literal_record = literal_store.record_verified_literal_action(
                device_id=DEVICE,
                resolved_action=resolved(),
                before_scene=literal_before,
                after_scene=literal_after,
                hardware_receipt=receipt(),
                after_frames=surface_frames(),
            )
            self.assertEqual("unknown", literal_record.app_id)
            self.assertEqual("input_field_1", literal_record.input_field_id)

            wrong_after = json.loads(json.dumps(literal_after))
            wrong_after["elements"][0]["states"][
                "input_field_id"
            ] = "input_field_2"
            with self.assertRaises(InputValueLineageError):
                TypedInputLineageStore(Path(literal_temp) / "wrong").record_verified_literal_action(
                    device_id=DEVICE,
                    resolved_action=resolved(),
                    before_scene=literal_before,
                    after_scene=wrong_after,
                    hardware_receipt=receipt(),
                    after_frames=surface_frames(),
                )

    def test_pending_ime_candidate_lineage_recovers_exact_committed_cue(self) -> None:
        before, action = ime_candidate_case()
        record = build_pending_ime_candidate_lineage(
            device_id=DEVICE,
            resolved_action=action,
            before_scene=before,
            hardware_receipt=receipt(),
        )
        self.assertEqual("pending_verified_ime_candidate_action", record.source)
        self.assertEqual("loopok", record.exact_value)
        self.assertTrue(
            record.matches_pending_input_state_surface(
                device_id=DEVICE,
                app_id="sample.app",
                screen_id="editor",
                input_bounds=(0.14, 0.54, 0.70, 0.61),
                input_field_id="input_field_1",
                now_epoch=record.recorded_at_epoch,
            )
        )
        self.assertTrue(
            record.matches_pending_input_state_surface(
                device_id=DEVICE,
                app_id="sample.app",
                screen_id="chat_input",
                input_bounds=(0.14, 0.54, 0.70, 0.61),
                input_field_id="input_field_1",
                now_epoch=record.recorded_at_epoch,
            )
        )
        self.assertFalse(
            record.matches_pending_input_state_surface(
                device_id=DEVICE,
                app_id="sample.app",
                screen_id="chat_input",
                input_bounds=(0.14, 0.54, 0.70, 0.61),
                input_field_id="input_field_2",
                now_epoch=record.recorded_at_epoch,
            )
        )

        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_exact_text",
                    "objective": "提交唯一完整候选到当前输入框",
                    "constraints": ["不得发送"],
                    "completion_conditions": ["输入框逐字等于 loopok"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "input_text": "loopok",
                        "active_input_transaction_text": "loopok",
                        "active_input_field_id": "input_field_1",
                        "active_input_multiline": False,
                    },
                }
            }
        }
        projected = _apply_input_structure_audit(
            UIScene.from_dict(before),
            ime_commit_audit_raw(),
            fingerprint="after-ime-candidate-fp",
            goal_context=context,
            ledger_input_value="loopok",
            verified_input_lineage=record,
            device_id=DEVICE,
            lineage_frame=surface_frame(),
        )
        field = projected.get_element("local_audited_input_1")
        self.assertEqual("loopok", field.states["value"])
        self.assertEqual("input_field_1", field.states["input_field_id"])
        self.assertTrue(
            any("应用输入框当前文字：loopok" in item for item in field.evidence)
        )

        drifted_before = json.loads(json.dumps(before))
        drifted_before["screen_id"] = "chat_input"
        projected_after_screen_name_drift = _apply_input_structure_audit(
            UIScene.from_dict(drifted_before),
            ime_commit_audit_raw(),
            fingerprint="after-ime-candidate-screen-name-drift",
            goal_context=context,
            ledger_input_value="",
            verified_input_lineage=record,
            device_id=DEVICE,
            lineage_frame=surface_frame(),
        )
        drifted_field = projected_after_screen_name_drift.get_element(
            "local_audited_input_1"
        )
        self.assertEqual("loopok", drifted_field.states["value"])
        self.assertEqual("input_field_1", drifted_field.states["input_field_id"])

        wrong_field_context = json.loads(json.dumps(context))
        wrong_field_context["entities"]["active_subgoal_visual_context"][
            "goal_entities"
        ]["active_input_field_id"] = "input_field_2"
        rejected_after_screen_name_drift = _apply_input_structure_audit(
            UIScene.from_dict(drifted_before),
            ime_commit_audit_raw(),
            fingerprint="after-ime-candidate-wrong-field",
            goal_context=wrong_field_context,
            ledger_input_value="",
            verified_input_lineage=record,
            device_id=DEVICE,
            lineage_frame=surface_frame(),
        )
        self.assertFalse(
            any(
                element.meaning == "application_text_input"
                and element.states.get("value") == "loopok"
                for element in rejected_after_screen_name_drift.elements
            )
        )

        for cue in ("", "loopo", "loopokx"):
            with self.subTest(cue=cue):
                rejected = _apply_input_structure_audit(
                    UIScene.from_dict(before),
                    ime_commit_audit_raw(cue=cue),
                    fingerprint="after-ime-candidate-rejected",
                    goal_context=context,
                    ledger_input_value="loopok",
                    verified_input_lineage=record,
                    device_id=DEVICE,
                    lineage_frame=surface_frame(),
                )
                self.assertFalse(
                    any(
                        element.meaning == "application_text_input"
                        and element.states.get("value") == "loopok"
                        for element in rejected.elements
                    )
                )

    def test_pending_ime_candidate_lineage_accepts_typed_unknown_app_and_prefix(self) -> None:
        cases = (
            {
                "prior": "",
                "segment": "你好",
                "pinyin": "nihao",
            },
            {
                "prior": "你好\n",
                "segment": "世界",
                "pinyin": "shijie",
            },
        )
        for case in cases:
            with self.subTest(case=case):
                before, action = ime_candidate_case(
                    **case,
                    app_id="unknown",
                    screen_id="multiline_input_acceptance",
                    bounds=(0.14, 0.27, 0.86, 0.45),
                )
                record = build_pending_ime_candidate_lineage(
                    device_id=DEVICE,
                    resolved_action=action,
                    before_scene=before,
                    hardware_receipt=receipt(),
                )
                self.assertEqual(case["prior"] + case["segment"], record.exact_value)
                self.assertEqual("unknown", record.app_id)
                self.assertEqual("input_field_1", record.input_field_id)
                self.assertTrue(
                    record.matches_typed_context(
                        device_id=DEVICE,
                        app_id="unknown",
                        screen_id="multiline_input_acceptance",
                        input_field_id="input_field_1",
                    )
                )
                self.assertFalse(
                    record.matches_typed_context(
                        device_id=DEVICE,
                        app_id="unknown",
                        screen_id="multiline_input_acceptance",
                        input_field_id="input_field_2",
                    )
                )

    def test_pending_ime_candidate_lineage_revokes_residual_prediction_preedit(self) -> None:
        cases = (
            {
                "prior": "",
                "segment": "你好",
                "pinyin": "nihao",
                "caret_marker": "|",
            },
            {
                "prior": "你好\n",
                "segment": "世界",
                "pinyin": "shijie",
                "caret_marker": "｜",
            },
            {
                "prior": "",
                "segment": "验收",
                "pinyin": "yanshou",
                "caret_marker": "|",
                "application_text": True,
            },
            {
                "prior": "",
                "segment": "你好",
                "pinyin": "nihao",
                "caret_marker": "|",
                "visible_editable_cues": ["bordered input area"],
            },
            {
                "prior": "",
                "segment": "aaazjie",
                "pinyin": "aaazjie",
                "caret_marker": "|",
                "input_mode": "direct_latin",
            },
            {
                "prior": "first\n",
                "segment": "second",
                "pinyin": "second",
                "caret_marker": "|",
                "input_mode": "direct_latin",
            },
        )
        records: list[tuple[dict, TypedInputLineage, dict, str]] = []
        for case in cases:
            with self.subTest(case=case):
                before, action = ime_candidate_case(
                    prior=case["prior"],
                    segment=case["segment"],
                    pinyin=case["pinyin"],
                    app_id="unknown",
                    screen_id="multiline_input_acceptance",
                    bounds=(0.14, 0.27, 0.86, 0.45),
                    input_mode=case.get("input_mode", "chinese_pinyin"),
                )
                record = build_pending_ime_candidate_lineage(
                    device_id=DEVICE,
                    resolved_action=action,
                    before_scene=before,
                    hardware_receipt=receipt(),
                )
                expected = case["prior"] + case["segment"]
                context = {
                    "entities": {
                        "active_subgoal_visual_context": {
                            "subgoal_id": "input_exact_text",
                            "objective": "在正文输入框完成精确中文",
                            "constraints": ["不得发送"],
                            "completion_conditions": [f"输入框逐字等于 {expected}"],
                            "execution_class": "navigate",
                            "goal_entities": {
                                "input_text": expected,
                                "active_input_transaction_text": expected,
                                "active_input_field_id": "input_field_1",
                                "active_input_field_label": "正文",
                                "active_input_multiline": "\n" in expected,
                            },
                        }
                    }
                }
                projected = _apply_input_structure_audit(
                    UIScene.from_dict(before),
                    ime_prediction_commit_audit_raw(
                        exact_value=expected,
                        candidate_text=case["segment"],
                        caret_marker=case["caret_marker"],
                        application_text=(
                            expected if case.get("application_text") else None
                        ),
                        visible_editable_cues=case.get(
                            "visible_editable_cues"
                        ),
                        input_mode=case.get("input_mode", "chinese_pinyin"),
                    ),
                    fingerprint="after-ime-prediction-commit",
                    goal_context=context,
                    ledger_input_value=expected,
                    verified_input_lineage=record,
                    device_id=DEVICE,
                    lineage_frame=surface_frame(),
                )
                field = projected.get_element("local_audited_input_1")
                self.assertEqual(expected, field.states["value"])
                self.assertNotIn("ime_preedit_text", field.states)
                self.assertFalse(
                    any(
                        element.meaning == "ime_exact_candidate"
                        and element.states.get("goal_relevant") is True
                        for element in projected.elements
                    )
                )
                self.assertTrue(
                    any(
                        "残留预测栏未作为预编辑" in evidence
                        for evidence in field.evidence
                    )
                )
                records.append((before, record, context, expected))

        before, record, context, expected = records[0]
        for coarse_value, cues in (
            (None, []),
            ("", ["bordered input area"]),
            ("conflicting compact value", ["focus border", "caret"]),
        ):
            with self.subTest(
                authority="typed-ledger",
                coarse_value=coarse_value,
                cues=cues,
            ):
                projected = _apply_input_structure_audit(
                    UIScene.from_dict(before),
                    ime_prediction_commit_audit_raw(
                        exact_value=expected,
                        candidate_text="你好",
                        visible_editable_cues=cues,
                    ),
                    fingerprint="after-ime-ledger-authority",
                    goal_context=context,
                    ledger_input_value=coarse_value,
                    verified_input_lineage=record,
                    device_id=DEVICE,
                    lineage_frame=surface_frame(),
                )
                field = projected.get_element("local_audited_input_1")
                self.assertEqual(expected, field.states["value"])
                self.assertNotIn("ime_preedit_text", field.states)

        negative_cases = (
            {
                "name": "wrong_preedit",
                "raw": ime_prediction_commit_audit_raw(
                    exact_value=expected,
                    candidate_text="你好",
                    preedit_text="错误",
                ),
                "lineage": record,
                "context": context,
            },
            {
                "name": "missing_lineage",
                "raw": ime_prediction_commit_audit_raw(
                    exact_value=expected,
                    candidate_text="你好",
                ),
                "lineage": None,
                "context": context,
            },
            {
                "name": "wrong_application_text",
                "raw": ime_prediction_commit_audit_raw(
                    exact_value=expected,
                    candidate_text="你好",
                    application_text=expected + "错",
                ),
                "lineage": record,
                "context": context,
            },
        )
        wrong_field_context = json.loads(json.dumps(context))
        wrong_field_context["entities"]["active_subgoal_visual_context"][
            "goal_entities"
        ]["active_input_field_id"] = "input_field_2"
        negative_cases += (
            {
                "name": "wrong_typed_field",
                "raw": ime_prediction_commit_audit_raw(
                    exact_value=expected,
                    candidate_text="你好",
                ),
                "lineage": record,
                "context": wrong_field_context,
            },
        )
        for case in negative_cases:
            with self.subTest(case=case["name"]):
                rejected = _apply_input_structure_audit(
                    UIScene.from_dict(before),
                    case["raw"],
                    fingerprint="after-ime-prediction-rejected",
                    goal_context=case["context"],
                    ledger_input_value=expected,
                    verified_input_lineage=case["lineage"],
                    device_id=DEVICE,
                    lineage_frame=surface_frame(),
                )
                self.assertFalse(
                    any(
                        element.meaning == "application_text_input"
                        and element.states.get("value") == expected
                        for element in rejected.elements
                    )
                )

    def test_pending_ime_candidate_lineage_rejects_identity_and_candidate_drift(self) -> None:
        before, action = ime_candidate_case()
        mutations = []
        missing_field = json.loads(json.dumps(before))
        missing_field["elements"][0]["states"].pop("input_field_id")
        mutations.append((missing_field, action))
        wrong_label = json.loads(json.dumps(before))
        wrong_label["elements"][1]["label"] = "different"
        mutations.append((wrong_label, action))
        duplicate = json.loads(json.dumps(before))
        duplicate_candidate = json.loads(json.dumps(duplicate["elements"][1]))
        duplicate_candidate["element_id"] = "candidate-duplicate"
        duplicate["elements"].append(duplicate_candidate)
        mutations.append((duplicate, action))
        wrong_expected = json.loads(json.dumps(action))
        wrong_expected["expected_input_value"] = "loopokx"
        mutations.append((before, wrong_expected))
        for candidate_before, candidate_action in mutations:
            with self.subTest(action=candidate_action), self.assertRaises(
                InputValueLineageError
            ):
                build_pending_ime_candidate_lineage(
                    device_id=DEVICE,
                    resolved_action=candidate_action,
                    before_scene=candidate_before,
                    hardware_receipt=receipt(),
                )

    def test_pending_input_state_lineage_rejects_wrong_surface_value_and_expiry(self) -> None:
        before, action = state_switch_case()
        record = build_pending_input_state_lineage(
            device_id=DEVICE,
            resolved_action=action,
            before_scene=before,
            hardware_receipt=receipt(),
            recorded_at_epoch=1000.0,
        )
        cases = (
            {"device_id": "other", "app_id": "sample.app", "screen_id": "editor", "raw_value": PRIOR, "input_bounds": (0.13, 0.54, 0.69, 0.61), "now_epoch": 1000.0},
            {"device_id": DEVICE, "app_id": "other.app", "screen_id": "editor", "raw_value": PRIOR, "input_bounds": (0.13, 0.54, 0.69, 0.61), "now_epoch": 1000.0},
            {"device_id": DEVICE, "app_id": "unknown", "screen_id": "editor", "raw_value": PRIOR, "input_bounds": (0.13, 0.54, 0.69, 0.61), "now_epoch": 1000.0},
            {"device_id": DEVICE, "app_id": "sample.app", "screen_id": "other", "raw_value": PRIOR, "input_bounds": (0.13, 0.54, 0.69, 0.61), "now_epoch": 1000.0},
            {"device_id": DEVICE, "app_id": "sample.app", "screen_id": "editor", "raw_value": PRIOR + "x", "input_bounds": (0.13, 0.54, 0.69, 0.61), "now_epoch": 1000.0},
            {"device_id": DEVICE, "app_id": "sample.app", "screen_id": "editor", "raw_value": PRIOR, "input_bounds": (0.75, 0.1, 0.95, 0.2), "now_epoch": 1000.0},
            {"device_id": DEVICE, "app_id": "sample.app", "screen_id": "editor", "raw_value": PRIOR, "input_bounds": (0.13, 0.54, 0.69, 0.61), "now_epoch": 22601.0},
        )
        for case in cases:
            with self.subTest(case=case):
                self.assertFalse(record.matches_pending_input_state_value(**case))
                surface_case = dict(case)
                surface_case.pop("raw_value")
                if case["raw_value"] != PRIOR:
                    continue
                self.assertFalse(
                    record.matches_pending_input_state_surface(**surface_case)
                )

    def test_pending_input_state_lineage_rejects_non_state_or_mutating_actions(self) -> None:
        before, action = state_switch_case()
        invalid_actions = []
        ordinary = dict(action)
        ordinary["target_element_id"] = "input-1"
        invalid_actions.append(ordinary)
        literal = dict(action)
        literal["target_element_id"] = "state-switch-1"
        before_literal = json.loads(json.dumps(before))
        before_literal["elements"][1]["meaning"] = "input_exact_literal_key"
        invalid_actions.append((before_literal, literal))
        mutating = json.loads(json.dumps(action))
        mutating["expected_input_value"] = PRIOR + "2"
        mutating["expected_effect"]["element_state"]["states"]["value"] = PRIOR + "2"
        invalid_actions.append(mutating)
        extra_state = json.loads(json.dumps(action))
        extra_state["expected_effect"]["element_state"]["states"]["extra"] = True
        invalid_actions.append(extra_state)
        for candidate in invalid_actions:
            candidate_before, candidate_action = (
                candidate if isinstance(candidate, tuple) else (before, candidate)
            )
            with self.subTest(action=candidate_action):
                with self.assertRaises(InputValueLineageError):
                    build_pending_input_state_lineage(
                        device_id=DEVICE,
                        resolved_action=candidate_action,
                        before_scene=candidate_before,
                        hardware_receipt=receipt(),
                    )
        bad_receipt = receipt()
        bad_receipt["seller_event_barrier_confirmed"] = False
        with self.assertRaises(InputValueLineageError):
            build_pending_input_state_lineage(
                device_id=DEVICE,
                resolved_action=action,
                before_scene=before,
                hardware_receipt=bad_receipt,
            )

    def test_symbol_routing_switches_to_english_before_symbol_layout(self) -> None:
        current = "aaazjie"
        target = current + "？"
        goal = {
            "objective": f"让输入框逐字显示 {target}",
            "entities": {"input_text": target},
        }
        mode_switch = {
            "label": "中/英",
            "bounds": [820, 900, 960, 960],
            "confidence": 1.0,
            "current_mode": "chinese_pinyin",
            "target_mode": "direct_latin",
        }
        switches = [
            {
                "label": "123",
                "bounds": [20, 900, 160, 960],
                "confidence": 1.0,
                "current_layout": "qwerty",
                "target_layout": "numeric",
            },
            {
                "label": "！？#",
                "bounds": [180, 900, 340, 960],
                "confidence": 1.0,
                "current_layout": "qwerty",
                "target_layout": "symbol",
            },
        ]

        audited = _apply_input_structure_audit(
            UIScene.from_dict(scene(current, "symbol-mode-before")),
            keyboard_routing_audit_raw(
                value=current,
                layout="qwerty",
                input_mode="chinese_pinyin",
                mode_switch=mode_switch,
                layout_switches=switches,
            ),
            fingerprint="symbol-mode-before",
            goal_context=goal,
            ledger_input_value=current,
        )
        goal_elements = [
            element for element in audited.elements
            if element.states.get("goal_relevant") is True
        ]
        self.assertEqual(
            ["switch_keyboard_input_mode"],
            [element.meaning for element in goal_elements],
        )
        self.assertEqual(
            "direct_latin",
            goal_elements[0].states["target_mode"],
        )

    def test_symbol_and_digit_use_distinct_visible_layout_switches(self) -> None:
        current = "aaazjie"
        switches = [
            {
                "label": "123",
                "bounds": [20, 900, 160, 960],
                "confidence": 1.0,
                "current_layout": "qwerty",
                "target_layout": "numeric",
            },
            {
                "label": "！？#",
                "bounds": [180, 900, 340, 960],
                "confidence": 1.0,
                "current_layout": "qwerty",
                "target_layout": "symbol",
            },
        ]
        cases = (
            ("？", "symbol", "！？#"),
            ("1", "numeric", "123"),
        )
        for suffix, expected_layout, expected_label in cases:
            with self.subTest(suffix=suffix):
                target = current + suffix
                audited = _apply_input_structure_audit(
                    UIScene.from_dict(scene(current, f"layout-{expected_layout}")),
                    keyboard_routing_audit_raw(
                        value=current,
                        layout="qwerty",
                        input_mode="direct_latin",
                        layout_switches=switches,
                    ),
                    fingerprint=f"layout-{expected_layout}",
                    goal_context={
                        "objective": f"让输入框逐字显示 {target}",
                        "entities": {"input_text": target},
                    },
                    ledger_input_value=current,
                )
                layout_element = audited.get_element(
                    "local_audited_keyboard_layout_switch_1"
                )
                self.assertEqual(expected_label, layout_element.label)
                self.assertEqual(
                    expected_layout,
                    layout_element.states["target_layout"],
                )

    def test_symbol_routing_never_uses_123_as_an_implicit_hop(self) -> None:
        current = "aaazjie"
        target = current + "？"
        audited = _apply_input_structure_audit(
            UIScene.from_dict(scene(current, "symbol-no-switch")),
            keyboard_routing_audit_raw(
                value=current,
                layout="qwerty",
                input_mode="direct_latin",
                layout_switches=[
                    {
                        "label": "123",
                        "bounds": [20, 900, 160, 960],
                        "confidence": 1.0,
                        "current_layout": "qwerty",
                        "target_layout": "numeric",
                    }
                ],
            ),
            fingerprint="symbol-no-switch",
            goal_context={
                "objective": f"让输入框逐字显示 {target}",
                "entities": {"input_text": target},
            },
            ledger_input_value=current,
        )
        self.assertFalse(
            any(
                element.meaning == "switch_keyboard_layout"
                and element.states.get("goal_relevant") is True
                for element in audited.elements
            )
        )

    def test_symbol_layout_exposes_only_the_exact_requested_character(self) -> None:
        current = "aaazjie"
        target = current + "？"
        audited = _apply_input_structure_audit(
            UIScene.from_dict(scene(current, "symbol-exact-key")),
            keyboard_routing_audit_raw(
                value=current,
                layout="symbol",
                input_mode="direct_latin",
                literal="？",
            ),
            fingerprint="symbol-exact-key",
            goal_context={
                "objective": f"让输入框逐字显示 {target}",
                "entities": {"input_text": target},
            },
            ledger_input_value=current,
        )
        literal = audited.get_element("local_audited_literal_key_1")
        self.assertEqual("？", literal.label)
        self.assertEqual("？", literal.states["key_value"])
        self.assertEqual(target, literal.states["expected_input_value"])

    def test_state_switch_live_audit_replay_preserves_exact_value_and_next_key(self) -> None:
        before, action = state_switch_case()
        record = build_pending_input_state_lineage(
            device_id=DEVICE,
            resolved_action=action,
            before_scene=before,
            hardware_receipt=receipt(),
        )
        audited = _apply_input_structure_audit(
            UIScene.from_dict(before),
            state_switch_audit_raw(),
            fingerprint="after-state-fp",
            goal_context={
                "objective": f"让输入框逐字显示 {PRIOR}2",
                "entities": {"input_text": PRIOR + "2"},
            },
            verified_input_lineage=record,
            device_id=DEVICE,
            lineage_frame=surface_frame(),
        )
        input_element = audited.get_element("local_audited_input_1")
        next_key = audited.get_element("local_audited_literal_key_1")
        self.assertEqual(PRIOR, input_element.states["value"])
        self.assertEqual(PRIOR, next_key.states["prior_input_value"])
        self.assertEqual(PRIOR + "2", next_key.states["expected_input_value"])
        self.assertTrue(any("输入状态切换后" in item for item in input_element.evidence))

        preedit_only = json.loads(state_switch_audit_raw())
        preedit_only["application_inputs"][0]["visible_editable_cues"] = [
            "caret",
            "underlined text",
        ]
        audited_preedit_only = _apply_input_structure_audit(
            UIScene.from_dict(before),
            json.dumps(preedit_only, ensure_ascii=False),
            fingerprint="after-state-fp-preedit-only",
            goal_context={
                "objective": f"让输入框逐字显示 {PRIOR}2",
                "entities": {"input_text": PRIOR + "2"},
            },
            verified_input_lineage=record,
            device_id=DEVICE,
            lineage_frame=surface_frame(),
        )
        self.assertEqual(
            PRIOR,
            audited_preedit_only.get_element(
                "local_audited_input_1"
            ).states["value"],
        )

    def test_state_switch_live_audit_requires_bound_lineage_and_exact_visible_cue(self) -> None:
        before, action = state_switch_case()
        record = build_pending_input_state_lineage(
            device_id=DEVICE,
            resolved_action=action,
            before_scene=before,
            hardware_receipt=receipt(),
        )
        goal = {
            "objective": f"让输入框逐字显示 {PRIOR}2",
            "entities": {"input_text": PRIOR + "2"},
        }
        for lineage, cue in ((None, PRIOR), (record, PRIOR + "x")):
            with self.subTest(lineage=lineage is not None, cue=cue):
                projected = _apply_input_structure_audit(
                    UIScene.from_dict(before),
                    state_switch_audit_raw(cue=cue),
                    fingerprint="after-state-fp",
                    goal_context=goal,
                    verified_input_lineage=lineage,
                    device_id=DEVICE,
                    lineage_frame=surface_frame(),
                )
                self.assertFalse(
                    any(
                        element.meaning == "input_exact_literal_key"
                        and element.states.get("goal_relevant") is True
                        for element in projected.elements
                    )
                )
        without_cue = _apply_input_structure_audit(
            UIScene.from_dict(before),
            state_switch_audit_raw(cue=""),
            fingerprint="after-state-fp",
            goal_context=goal,
            verified_input_lineage=record,
            device_id=DEVICE,
            lineage_frame=surface_frame(),
        )
        self.assertFalse(
            any(
                element.meaning == "input_exact_literal_key"
                and element.states.get("goal_relevant") is True
                for element in without_cue.elements
            )
        )


    def test_pending_direct_text_lineage_never_persists_or_accepts_pinyin(self) -> None:
        action = resolved_text()
        pending = build_pending_text_lineage(
            device_id=DEVICE,
            resolved_action=action,
            before_scene=scene("", "before-fp"),
            recorded_at_epoch=1000.0,
        )
        self.assertEqual("pending_verified_text_action", pending.source)
        self.assertEqual((), pending.surface_descriptors)
        self.assertTrue(
            lineage_matches_visual(
                pending,
                device_id=DEVICE,
                app_id="sample.app",
                screen_id="editor",
                raw_value="long\ninput",
                now_epoch=1000.0,
            )
        )
        pinyin = dict(action)
        pinyin["input_method"] = "chinese_pinyin"
        with self.assertRaises(InputValueLineageError):
            build_pending_text_lineage(
                device_id=DEVICE,
                resolved_action=pinyin,
                before_scene=scene("", "before-fp"),
            )
        unknown_untyped = scene("", "before-fp")
        unknown_untyped["foreground_app_id"] = "unknown"
        unknown_untyped["app_id"] = "unknown"
        with self.assertRaises(InputValueLineageError):
            build_pending_text_lineage(
                device_id=DEVICE,
                resolved_action=action,
                before_scene=unknown_untyped,
            )

    def test_pending_chinese_preedit_lineage_keeps_typed_prefix_across_newline(
        self,
    ) -> None:
        cases = (
            ("", "你好", "nihao", "你好\n世界"),
            ("你好\n", "世界", "shijie", "你好\n世界"),
        )
        for prior, fragment, pinyin, target_text in cases:
            with self.subTest(prior=prior, fragment=fragment):
                before = scene(prior, "before-chinese-preedit")
                before["foreground_app_id"] = "unknown"
                before["app_id"] = "unknown"
                before["elements"][0]["states"].update(
                    {
                        "input_field_id": "input_field_1",
                        "input_multiline": True,
                        "keyboard_layout": "qwerty",
                        "keyboard_input_mode": "chinese_pinyin",
                        "keyboard_geometry": {"anchors": {"q": [0.1, 0.8]}},
                    }
                )
                pending = build_pending_chinese_preedit_lineage(
                    device_id=DEVICE,
                    resolved_action=resolved_chinese_preedit(
                        prior=prior,
                        fragment=fragment,
                        pinyin=pinyin,
                        target_text=target_text,
                    ),
                    before_scene=before,
                    recorded_at_epoch=1000.0,
                )
                self.assertEqual(
                    "pending_verified_chinese_preedit_action",
                    pending.source,
                )
                self.assertEqual(prior + fragment, pending.exact_value)
                self.assertEqual("input_field_1", pending.input_field_id)
                self.assertTrue(
                    pending.matches_typed_context(
                        device_id=DEVICE,
                        app_id="unknown",
                        screen_id="editor",
                        input_field_id="input_field_1",
                        now_epoch=1000.0,
                    )
                )

    def test_pending_chinese_preedit_lineage_rejects_wrong_authority_or_mode(
        self,
    ) -> None:
        before = scene("", "before-chinese-preedit")
        before["elements"][0]["states"].update(
            {
                "input_field_id": "input_field_1",
                "keyboard_layout": "qwerty",
                "keyboard_input_mode": "chinese_pinyin",
                "keyboard_geometry": {"anchors": {"q": [0.1, 0.8]}},
            }
        )
        wrong_pinyin = resolved_chinese_preedit(pinyin="shijie")
        wrong_effect = resolved_chinese_preedit()
        wrong_effect["expected_effect"]["element_state"]["states"][
            "ime_exact_candidate_text"
        ] = "您好"
        direct = resolved_chinese_preedit()
        direct["input_method"] = "direct_latin"
        for action in (wrong_pinyin, wrong_effect, direct):
            with self.subTest(action=action):
                with self.assertRaises(InputValueLineageError):
                    build_pending_chinese_preedit_lineage(
                        device_id=DEVICE,
                        resolved_action=action,
                        before_scene=before,
                    )

    def test_pending_text_preedit_preserves_same_typed_multiline_prefix(self) -> None:
        prior = "first\n"
        fragment = "second"
        expected = prior + fragment
        before = scene(prior, "before-fp")
        before["elements"][0]["states"].update(
            {
                "input_field_id": "input_field_1",
                "input_multiline": True,
                "keyboard_layout": "qwerty",
                "keyboard_input_mode": "direct_latin",
            }
        )
        before["foreground_app_id"] = "unknown"
        before["app_id"] = "unknown"
        pending = build_pending_text_lineage(
            device_id=DEVICE,
            resolved_action=resolved_text(prior=prior, fragment=fragment),
            before_scene=before,
        )
        audit = {
            "protocol_version": INPUT_STRUCTURE_AUDIT_VERSION,
            "application_inputs": [
                {
                    "structure_id": "app-input-1",
                    "bounds": [130, 540, 690, 610],
                    "fully_visible": True,
                    "text": "",
                    "placeholder": "",
                    "visible_editable_cues": [
                        "正文",
                        "complete border",
                        "focus highlight",
                    ],
                    "caret_line_index": 1,
                    "confidence": 1.0,
                    "right_button": None,
                }
            ],
            "ime_preedit_regions": [
                {
                    "region_id": "ime-preedit-1",
                    "bounds": [150, 565, 500, 600],
                    "text": fragment,
                    "confidence": 1.0,
                    "candidates": [
                        {
                            "text": fragment,
                            "bounds": [100, 650, 220, 690],
                            "confidence": 1.0,
                            "fully_visible": True,
                        },
                        {
                            "text": "secondary",
                            "bounds": [240, 650, 430, 690],
                            "confidence": 1.0,
                            "fully_visible": True,
                        },
                    ],
                }
            ],
            "keyboard": {
                "visible": True,
                "bounds": [0, 620, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "case_mode": "lower",
                "qwerty_anchors": {
                    "q": [122, 735],
                    "p": [880, 735],
                    "a": [164, 810],
                    "l": [838, 810],
                    "z": [248, 885],
                    "m": [754, 885],
                    "backspace": [880, 885],
                },
                "mode_switch": None,
                "backspace_key": None,
                "enter_key": None,
                "case_switch": None,
                "literal_keys": [],
                "layout_switches": [],
            },
        }

        def goal(field_id: str = "input_field_1", text: str = expected) -> dict:
            return {
                "entities": {
                    "active_subgoal_visual_context": {
                        "subgoal_id": "input_exact_text",
                        "objective": "输入精确多行文字",
                        "constraints": [],
                        "completion_conditions": [],
                        "execution_class": "navigate",
                        "goal_entities": {
                            "input_text": text,
                            "active_input_transaction_text": text,
                            "active_input_field_id": field_id,
                            "active_input_multiline": True,
                        },
                    }
                }
            }

        base_payload = scene(expected, "after-fp")
        base_payload["foreground_app_id"] = "unknown"
        base_payload["app_id"] = "unknown"
        base = UIScene.from_dict(base_payload)
        projected = _apply_input_structure_audit(
            base,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="after-fp",
            goal_context=goal(),
            ledger_input_value=expected,
            verified_input_lineage=pending,
            device_id=DEVICE,
        )
        input_element = projected.get_element("local_audited_input_1")
        candidate = projected.get_element("local_audited_ime_candidate_1")
        self.assertEqual(prior, input_element.states["value"])
        self.assertEqual(fragment, input_element.states["ime_preedit_text"])
        self.assertEqual(prior, candidate.states["prior_input_value"])
        self.assertEqual(expected, candidate.states["expected_input_value"])
        self.assertTrue(any("pending typed连续性" in item for item in input_element.evidence))

        candidate_mutations = {}
        missing_candidate = json.loads(json.dumps(audit, ensure_ascii=False))
        missing_candidate["ime_preedit_regions"][0]["candidates"] = []
        candidate_mutations["missing"] = missing_candidate
        wrong_candidate = json.loads(json.dumps(audit, ensure_ascii=False))
        wrong_candidate["ime_preedit_regions"][0]["candidates"][0][
            "text"
        ] = "different"
        candidate_mutations["wrong"] = wrong_candidate
        duplicate_candidate = json.loads(json.dumps(audit, ensure_ascii=False))
        duplicate_candidate["ime_preedit_regions"][0]["candidates"].append(
            dict(duplicate_candidate["ime_preedit_regions"][0]["candidates"][0])
        )
        candidate_mutations["duplicate"] = duplicate_candidate
        for name, changed_audit in candidate_mutations.items():
            with self.subTest(candidate=name), self.assertRaisesRegex(
                VisionAgentError,
                "唯一逐字候选几何",
            ):
                _apply_input_structure_audit(
                    base,
                    json.dumps(changed_audit, ensure_ascii=False),
                    fingerprint="after-fp",
                    goal_context=goal(),
                    ledger_input_value=expected,
                    verified_input_lineage=pending,
                    device_id=DEVICE,
                )

        for changed_goal in (
            (goal(field_id="other_field"), expected),
            (goal(text="first\nthird"), expected),
        ):
            changed_goal, changed_coarse = changed_goal
            with self.subTest(goal=changed_goal):
                rejected = _apply_input_structure_audit(
                    base,
                    json.dumps(audit, ensure_ascii=False),
                    fingerprint="after-fp",
                    goal_context=changed_goal,
                    ledger_input_value=changed_coarse,
                    verified_input_lineage=pending,
                    device_id=DEVICE,
                )
                self.assertEqual(
                    "",
                    rejected.get_element("local_audited_input_1").states["value"],
                )
                self.assertFalse(
                    any(
                        element.meaning == "ime_exact_candidate"
                        for element in rejected.elements
                    )
                )

        compact_conflict = _apply_input_structure_audit(
            base,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="after-fp",
            goal_context=goal(),
            ledger_input_value="first\nother",
            verified_input_lineage=pending,
            device_id=DEVICE,
        )
        compact_input = compact_conflict.get_element("local_audited_input_1")
        self.assertEqual(prior, compact_input.states["value"])
        self.assertEqual(fragment, compact_input.states["ime_preedit_text"])

    def test_direct_text_lineage_rejects_broken_chain_and_missing_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = self.make_store(temp)
            broken = resolved_text()
            broken["expected_input_value"] = "different"
            with self.assertRaises(InputValueLineageError):
                store.record_verified_text_action(
                    device_id=DEVICE,
                    resolved_action=broken,
                    before_scene=scene("", "before-fp"),
                    after_scene=scene("longinput", "after-fp"),
                    after_frames=surface_frames(),
                )
            with self.assertRaises(InputValueLineageError):
                store.record_verified_text_action(
                    device_id=DEVICE,
                    resolved_action=resolved_text(),
                    before_scene=scene("", "before-fp"),
                    after_scene=scene("longinput", "after-fp"),
                    after_frames=surface_frames()[:3],
                )

    def test_successful_clear_discards_only_that_device_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = self.make_store(temp)
            store.record_verified_text_action(
                device_id=DEVICE,
                resolved_action=resolved_text(),
                before_scene=scene("", "before-fp"),
                after_scene=scene("longinput", "after-fp"),
                after_frames=surface_frames(),
            )
            self.assertIsNotNone(store.load(DEVICE))
            store.discard(DEVICE)
            self.assertIsNone(store.load(DEVICE))
            store.discard(DEVICE)


    def test_lineage_canonicalizes_only_matching_visual_soft_wrap(self) -> None:
        record = TypedInputLineage(
            version=TYPED_INPUT_LINEAGE_VERSION,
            device_id=DEVICE,
            exact_value="livex",
            app_id="sample.app",
            screen_id="editor",
            input_meaning="application_text_input",
            input_field_id="input_field_1",
            input_bounds=(0.13, 0.54, 0.69, 0.61),
            before_fingerprint="before-fp",
            after_fingerprint="after-fp",
            action_digest="a" * 64,
            receipt_digest="b" * 64,
            surface_descriptors=tuple(
                describe_input_surface(
                    surface_frame(variation=index % 2),
                    (0.13, 0.54, 0.69, 0.61),
                )
                for index in range(4)
            ),
            recorded_at_epoch=time.time(),
            source="verified_live_literal_action",
        )
        base = UIScene.from_dict(
            scene(
                "li\nvex",
                "current-fp",
                app_id="arbitrary.model.name",
                screen_id="unrelated_model_screen_name",
            )
        )
        goal = {
            "objective": "让输入框显示 livex7",
            "entities": {"input_text": "livex7"},
        }
        audited = _apply_input_structure_audit(
            base,
            input_audit_raw("li\nvex", "7"),
            fingerprint="current-fp",
            goal_context=goal,
            verified_input_lineage=record,
            device_id=DEVICE,
            lineage_frame=surface_frame(variation=1),
        )
        input_element = audited.get_element("local_audited_input_1")
        key = audited.get_element("local_audited_literal_key_1")
        self.assertEqual(input_element.states["value"], "livex")
        self.assertEqual(key.states["prior_input_value"], "livex")
        self.assertEqual(key.states["expected_input_value"], "livex7")
        self.assertTrue(any("视觉折行转写" in item for item in input_element.evidence))

    def test_lineage_does_not_canonicalize_different_or_real_newline_value(self) -> None:
        record = TypedInputLineage(
            version=TYPED_INPUT_LINEAGE_VERSION,
            device_id=DEVICE,
            exact_value="line1line2",
            app_id="sample.app",
            screen_id="editor",
            input_meaning="application_text_input",
            input_field_id="input_field_1",
            input_bounds=(0.13, 0.54, 0.69, 0.61),
            before_fingerprint="before-fp",
            after_fingerprint="after-fp",
            action_digest="a" * 64,
            receipt_digest="b" * 64,
            surface_descriptors=tuple(
                describe_input_surface(
                    surface_frame(variation=index % 2),
                    (0.13, 0.54, 0.69, 0.61),
                )
                for index in range(4)
            ),
            recorded_at_epoch=time.time(),
            source="verified_live_literal_action",
        )
        base = UIScene.from_dict(scene("line1\nDIFFERENT", "current-fp"))
        audited = _apply_input_structure_audit(
            base,
            input_audit_raw("line1\nDIFFERENT", None),
            fingerprint="current-fp",
            goal_context={"objective": "保留真实换行", "entities": {}},
            verified_input_lineage=record,
            device_id=DEVICE,
        )
        self.assertEqual(
            audited.get_element("local_audited_input_1").states["value"],
            "line1\nDIFFERENT",
        )

    def test_multiline_field_never_collapses_real_trailing_newline_to_lineage(self) -> None:
        record = TypedInputLineage(
            version=TYPED_INPUT_LINEAGE_VERSION,
            device_id=DEVICE,
            exact_value="first",
            app_id="sample.app",
            screen_id="editor",
            input_meaning="application_text_input",
            input_field_id="input_field_1",
            input_bounds=(0.13, 0.54, 0.69, 0.61),
            before_fingerprint="before-fp",
            after_fingerprint="after-fp",
            action_digest="a" * 64,
            receipt_digest="b" * 64,
            surface_descriptors=tuple(
                describe_input_surface(
                    surface_frame(variation=index % 2),
                    (0.13, 0.54, 0.69, 0.61),
                )
                for index in range(4)
            ),
            recorded_at_epoch=time.time(),
            source="verified_live_literal_action",
        )
        base = UIScene.from_dict(scene("first\n", "current-fp"))
        goal = {
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

        audited = _apply_input_structure_audit(
            base,
            input_audit_raw("first\n", None),
            fingerprint="current-fp",
            goal_context=goal,
            verified_input_lineage=record,
            device_id=DEVICE,
            lineage_frame=surface_frame(variation=1),
        )

        self.assertEqual(
            "first\n",
            audited.get_element("local_audited_input_1").states["value"],
        )

    def test_pending_newline_requires_same_typed_field_exact_prior_and_next_caret_row(
        self,
    ) -> None:
        action, before, _after = newline_case()
        pending = build_pending_newline_lineage(
            device_id=DEVICE,
            resolved_action=action,
            before_scene=before,
            hardware_receipt=receipt(),
            recorded_at_epoch=1000.0,
        )
        matching = {
            "device_id": DEVICE,
            "app_id": "sample.app",
            "screen_id": "editor_input",
            "raw_value": "",
            "visible_editable_cues": ("first", "|"),
            "caret_line_index": 1,
            "input_bounds": (0.13, 0.56, 0.69, 0.63),
            "input_field_id": "input_field_1",
            "now_epoch": 1000.0,
        }
        self.assertTrue(
            lineage_matches_trailing_newline_cue(pending, **matching)
        )
        for changed in (
            {"caret_line_index": 0},
            {"input_field_id": "other_field"},
            {"visible_editable_cues": ("firstx", "|")},
            {"screen_id": "unrelated", "input_field_id": "other_field"},
        ):
            with self.subTest(changed=changed):
                self.assertFalse(
                    lineage_matches_trailing_newline_cue(
                        pending,
                        **{**matching, **changed}
                    )
                )

    def test_pending_and_verified_newline_accept_typed_unknown_app(self) -> None:
        for prior in ("first", "你好"):
            with self.subTest(prior=prior), tempfile.TemporaryDirectory() as temp:
                action, before, after = newline_case(prior)
                for candidate in (before, after):
                    candidate["foreground_app_id"] = "unknown"
                    candidate["app_id"] = "unknown"

                pending = build_pending_newline_lineage(
                    device_id=DEVICE,
                    resolved_action=action,
                    before_scene=before,
                    hardware_receipt=receipt(),
                    recorded_at_epoch=1000.0,
                )
                self.assertEqual(prior + "\n", pending.exact_value)
                self.assertEqual("unknown", pending.app_id)
                self.assertEqual("input_field_1", pending.input_field_id)
                self.assertTrue(
                    lineage_matches_trailing_newline_cue(
                        pending,
                        device_id=DEVICE,
                        app_id="unknown",
                        screen_id="editor",
                        raw_value="",
                        visible_editable_cues=(prior, "caret"),
                        caret_line_index=1,
                        input_bounds=(0.13, 0.54, 0.69, 0.61),
                        input_field_id="input_field_1",
                        now_epoch=1000.0,
                    )
                )

                store = TypedInputLineageStore(
                    Path(temp),
                    clock=lambda: 1000.0,
                )
                verified = store.record_verified_newline_action(
                    device_id=DEVICE,
                    resolved_action=action,
                    before_scene=before,
                    after_scene=after,
                    hardware_receipt=receipt(),
                    after_frames=surface_frames(),
                )
                self.assertEqual("unknown", verified.app_id)
                self.assertTrue(
                    verified.matches_typed_context(
                        device_id=DEVICE,
                        app_id="unknown",
                        screen_id="editor",
                        input_field_id="input_field_1",
                        now_epoch=1000.0,
                    )
                )

    def test_pending_newline_rejects_unknown_app_without_typed_field(self) -> None:
        action, before, _after = newline_case()
        before["foreground_app_id"] = "unknown"
        before["app_id"] = "unknown"
        before["elements"][0]["states"].pop("input_field_id")
        before["elements"][1]["states"]["input_field_id"] = "unknown"
        with self.assertRaises(InputValueLineageError):
            build_pending_newline_lineage(
                device_id=DEVICE,
                resolved_action=action,
                before_scene=before,
                hardware_receipt=receipt(),
            )

    def test_pending_newline_projects_exact_value_into_same_typed_field(self) -> None:
        action, before, _after = newline_case()
        pending = build_pending_newline_lineage(
            device_id=DEVICE,
            resolved_action=action,
            before_scene=before,
            hardware_receipt=receipt(),
            recorded_at_epoch=time.time(),
        )
        audit = {
            "protocol_version": INPUT_STRUCTURE_AUDIT_VERSION,
            "application_inputs": [
                {
                    "structure_id": "app-input-1",
                    "bounds": [130, 560, 690, 630],
                    "fully_visible": True,
                    "text": "",
                    "placeholder": "",
                    "visible_editable_cues": ["first", "|"],
                    "caret_line_index": 1,
                    "confidence": 1.0,
                    "right_button": None,
                }
            ],
            "ime_preedit_regions": [],
            "keyboard": {
                "visible": True,
                "bounds": [0, 700, 1000, 1000],
                "layout": "symbol",
                "input_mode": "direct_latin",
                "mode_switch": None,
            },
        }
        goal = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_exact_text",
                    "objective": "输入精确文字",
                    "constraints": [],
                    "completion_conditions": [],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "input_text": "first\n",
                        "active_input_transaction_text": "first\n",
                        "active_input_field_id": "input_field_1",
                        "active_input_multiline": True,
                    },
                }
            }
        }
        base = UIScene.from_dict(scene("", "newline-current", screen_id="editor_input"))
        projected = _apply_input_structure_audit(
            base,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="newline-current",
            goal_context=goal,
            verified_input_lineage=pending,
            device_id=DEVICE,
            lineage_frame=surface_frame(variation=1),
        )
        input_element = projected.get_element("local_audited_input_1")
        self.assertEqual("first\n", input_element.states["value"])
        self.assertIs(input_element.states["verified_trailing_newline"], True)
        self.assertTrue(
            any("已验证换行动作" in item for item in input_element.evidence)
        )

        audit["application_inputs"][0]["caret_line_index"] = 0
        rejected = _apply_input_structure_audit(
            base,
            json.dumps(audit, ensure_ascii=False),
            fingerprint="newline-current",
            goal_context=goal,
            verified_input_lineage=pending,
            device_id=DEVICE,
            lineage_frame=surface_frame(variation=1),
        )
        self.assertEqual(
            "",
            rejected.get_element("local_audited_input_1").states["value"],
        )

    def test_verified_newline_persists_only_after_exact_post_action_proof(self) -> None:
        action, before, after = newline_case()
        with tempfile.TemporaryDirectory() as temp:
            store = TypedInputLineageStore(Path(temp), clock=lambda: 1000.0)
            record = store.record_verified_newline_action(
                device_id=DEVICE,
                resolved_action=action,
                before_scene=before,
                after_scene=after,
                hardware_receipt=receipt(),
                after_frames=surface_frames(),
            )
            self.assertEqual("first\n", record.exact_value)
            self.assertTrue(
                lineage_matches_trailing_newline_cue(
                    record,
                    device_id=DEVICE,
                    app_id="sample.app",
                    screen_id="editor_input",
                    raw_value="",
                    visible_editable_cues=("first", "|"),
                    caret_line_index=1,
                    input_bounds=(0.13, 0.54, 0.69, 0.61),
                    input_field_id="input_field_1",
                    current_frame=surface_frame(variation=1),
                    now_epoch=1000.0,
                )
            )
            broken_after = json.loads(json.dumps(after))
            broken_after["elements"][0]["states"].pop(
                "verified_trailing_newline"
            )
            with self.assertRaises(InputValueLineageError):
                store.record_verified_newline_action(
                    device_id=DEVICE,
                    resolved_action=action,
                    before_scene=before,
                    after_scene=broken_after,
                    hardware_receipt=receipt(),
                    after_frames=surface_frames(),
                )


if __name__ == "__main__":
    unittest.main()
