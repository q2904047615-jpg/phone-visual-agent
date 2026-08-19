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
    build_pending_literal_lineage,
)
from ui_scene import UIScene


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
