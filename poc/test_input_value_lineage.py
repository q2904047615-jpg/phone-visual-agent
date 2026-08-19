from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from generic_scene_observer import _apply_input_structure_audit
from input_value_lineage import (
    InputValueLineageError,
    TYPED_INPUT_LINEAGE_VERSION,
    TypedInputLineage,
    TypedInputLineageStore,
    build_pending_literal_lineage,
)
from ui_scene import UIScene


DEVICE = "device-test-01"
PRIOR = "long2026:123@"
EXPECTED = PRIOR + "7"
RAW_AFTER = "long2026:\n123@7"


def scene(value: str, fingerprint: str, *, bounds=(0.13, 0.54, 0.69, 0.61)) -> dict:
    return {
        "protocol_version": "2026-08-14-ui-scene-v3",
        "foreground_app_id": "sample.app",
        "app_id": "sample.app",
        "screen_id": "editor",
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
                path.write_bytes(b"evidence")
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
            recorded_at_epoch=time.time(),
            source="verified_live_literal_action",
        )
        base = UIScene.from_dict(scene("li\nvex", "current-fp"))
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
