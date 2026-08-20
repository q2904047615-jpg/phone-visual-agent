from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from generic_scene_observer import _apply_input_structure_audit
from input_value_lineage import (
    InputValueLineageError,
    TYPED_INPUT_LINEAGE_VERSION,
    TypedInputLineage,
    TypedInputLineageStore,
    _surface_descriptor,
    build_pending_input_state_lineage,
    build_pending_literal_lineage,
    build_pending_text_lineage,
)
from ui_scene import UIScene
from vision_agent import VisionAgentError


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
                "geometry_audit_source": "element_geometry_audit",
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


def input_audit_raw(value: str, literal: str | None = "x") -> str:
    return json.dumps(
        {
            "protocol_version": "2026-08-18-input-structure-audit-v7",
            "application_inputs": [
                {
                    "structure_id": "input-1",
                    "bounds": [130, 540, 690, 610],
                    "fully_visible": True,
                    "text": value,
                    "placeholder": "",
                    "visible_editable_cues": ["cursor"],
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
            "protocol_version": "2026-08-18-input-structure-audit-v7",
            "application_inputs": [
                {
                    "structure_id": "app-input-1",
                    "bounds": [130, 540, 690, 600],
                    "fully_visible": True,
                    "text": "",
                    "placeholder": "",
                    "visible_editable_cues": ([] if not cue else [cue]),
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


class TypedInputLineageTests(unittest.TestCase):
    def make_store(self, root: str, now: float = 1000.0) -> TypedInputLineageStore:
        return TypedInputLineageStore(Path(root), clock=lambda: now)

    def test_verified_literal_action_round_trip_and_visual_match(self) -> None:
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
            self.assertEqual(record.exact_value, EXPECTED)
            loaded = store.match_visual(
                device_id=DEVICE,
                app_id="sample.app",
                screen_id="editor",
                raw_value=RAW_AFTER,
                input_bounds=(0.13, 0.54, 0.69, 0.61),
            )
            self.assertEqual(loaded, record)

    def test_wrong_device_surface_value_geometry_and_expiry_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            now = [1000.0]
            store = TypedInputLineageStore(
                Path(temp), ttl_seconds=10, clock=lambda: now[0]
            )
            store.record_verified_literal_action(
                device_id=DEVICE,
                resolved_action=resolved(),
                before_scene=before_scene(),
                after_scene=scene(RAW_AFTER, "after-fp"),
                hardware_receipt=receipt(),
                after_frames=surface_frames(),
            )
            cases = (
                {"device_id": "other", "app_id": "sample.app", "screen_id": "editor", "raw_value": RAW_AFTER},
                {"device_id": DEVICE, "app_id": "other.app", "screen_id": "editor", "raw_value": RAW_AFTER},
                {"device_id": DEVICE, "app_id": "sample.app", "screen_id": "other", "raw_value": RAW_AFTER},
                {"device_id": DEVICE, "app_id": "sample.app", "screen_id": "editor", "raw_value": RAW_AFTER + "x"},
                {"device_id": DEVICE, "app_id": "sample.app", "screen_id": "editor", "raw_value": RAW_AFTER, "input_bounds": (0.75, 0.1, 0.95, 0.2)},
            )
            for kwargs in cases:
                self.assertIsNone(store.match_visual(**kwargs))
            now[0] = 1011.0
            self.assertIsNone(
                store.match_visual(
                    device_id=DEVICE,
                    app_id="sample.app",
                    screen_id="editor",
                    raw_value=RAW_AFTER,
                )
            )

    def test_wrapped_input_height_change_keeps_same_surface_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = self.make_store(temp)
            store.record_verified_literal_action(
                device_id=DEVICE,
                resolved_action=resolved(),
                before_scene=before_scene(),
                after_scene=scene(RAW_AFTER, "after-fp"),
                hardware_receipt=receipt(),
                after_frames=surface_frames(),
            )
            matched = store.match_visual(
                device_id=DEVICE,
                app_id="sample.app",
                screen_id="editor",
                raw_value=RAW_AFTER,
                input_bounds=(0.13, 0.59, 0.68, 0.69),
            )
            self.assertIsNotNone(matched)
            self.assertIsNone(
                store.match_visual(
                    device_id=DEVICE,
                    app_id="sample.app",
                    screen_id="editor",
                    raw_value=RAW_AFTER,
                    input_bounds=(0.13, 0.64, 0.68, 0.74),
                )
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
                record.matches_visual(
                    device_id=DEVICE,
                    app_id="unknown",
                    screen_id="editor_composing",
                    raw_value=RAW_AFTER,
                    now_epoch=1000.0,
                )
            )
            self.assertFalse(
                record.matches_visual(
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
                record.matches_visual(
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
                record.matches_visual(
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
                record.matches_visual(
                    device_id=DEVICE,
                    app_id="arbitrary.model.name",
                    screen_id="unrelated_model_screen_name",
                    raw_value=RAW_AFTER,
                    input_bounds=(0.13, 0.59, 0.68, 0.69),
                    now_epoch=1000.0,
                )
            )
            self.assertFalse(
                record.matches_visual(
                    device_id=DEVICE,
                    app_id="unknown",
                    screen_id="unrelated_surface",
                    raw_value=RAW_AFTER,
                    now_epoch=1000.0,
                )
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
                        "geometry_audit_source": "element_geometry_audit",
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
            record.matches_visual(
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
                    record.matches_pending_input_state_value(
                        device_id=DEVICE,
                        app_id="sample.app",
                        screen_id="editor",
                        raw_value=PRIOR,
                        input_bounds=(0.13, 0.54, 0.69, 0.61),
                        now_epoch=1000.0,
                    )
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

    def test_verified_direct_text_action_persists_and_matches_soft_wrap(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = self.make_store(temp)
            action = resolved_text()
            record = store.record_verified_text_action(
                device_id=DEVICE,
                resolved_action=action,
                before_scene=scene("", "before-fp"),
                after_scene=scene("long\ninput", "after-fp"),
                after_frames=surface_frames(),
            )
            self.assertEqual("longinput", record.exact_value)
            self.assertEqual("verified_live_text_action", record.source)
            self.assertEqual(
                record,
                store.match_visual(
                    device_id=DEVICE,
                    app_id="sample.app",
                    screen_id="editor",
                    raw_value="long\ninput",
                    input_bounds=(0.13, 0.54, 0.69, 0.61),
                ),
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
            pending.matches_visual(
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

    def test_recover_persisted_execution_requires_four_existing_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = []
            for index in range(8):
                path = root / f"frame-{index}.jpg"
                surface_frame(variation=index % 2).save(path, format="JPEG")
                paths.append(str(path))
            payload = {
                "device_id": DEVICE,
                "history": [
                    {
                        "execution": {
                            "physical_actions": 1,
                            "resolved_action": resolved(),
                            "before_scene": before_scene(),
                            "after_scene": scene(RAW_AFTER, "after-fp"),
                            "hardware_receipt": receipt(),
                            "before_frame_paths": paths[:4],
                            "after_frame_paths": paths[4:],
                        }
                    }
                ],
            }
            session_path = root / "session.json"
            session_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            store = self.make_store(str(root / "state"))
            recovered = store.recover_from_session_file(session_path)
            self.assertEqual(recovered.source, "verified_persisted_literal_execution")
            Path(paths[-1]).unlink()
            with self.assertRaises(InputValueLineageError):
                store.recover_from_session_file(session_path)

    def test_lineage_canonicalizes_only_matching_visual_soft_wrap(self) -> None:
        record = TypedInputLineage(
            version=TYPED_INPUT_LINEAGE_VERSION,
            device_id=DEVICE,
            exact_value="livex",
            app_id="sample.app",
            screen_id="editor",
            input_meaning="application_text_input",
            input_bounds=(0.13, 0.54, 0.69, 0.61),
            before_fingerprint="before-fp",
            after_fingerprint="after-fp",
            action_digest="a" * 64,
            receipt_digest="b" * 64,
            surface_descriptors=tuple(
                _surface_descriptor(
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
            input_bounds=(0.13, 0.54, 0.69, 0.61),
            before_fingerprint="before-fp",
            after_fingerprint="after-fp",
            action_digest="a" * 64,
            receipt_digest="b" * 64,
            surface_descriptors=tuple(
                _surface_descriptor(
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


if __name__ == "__main__":
    unittest.main()
