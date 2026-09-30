from __future__ import annotations

import unittest
from types import SimpleNamespace

from agent.domain.canonical_action_protocol import (
    CanonicalActionProtocolError,
    bind_same_response_action,
    normalize_model_step_decision,
)
from agent.domain.universal_action_controller import UniversalActionController
from agent.domain.ui_scene import UIScene
from agent.infrastructure.generic_scene_observer import _single_step_response_format


class GenericActionWireProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = UIScene(
            app_id="sample.app", screen_id="sample", summary="画面证据只作诊断",
            elements=(), stable=True, confidence=1.0, fingerprint="f" * 64,
        )
        self.observation = SimpleNamespace(scene=self.scene)
        self.context = SimpleNamespace(revision=1, device_id="device-local-01", exact_input_text=None)

    def test_schema_exposes_only_generic_geometry_fields(self) -> None:
        schema = _single_step_response_format(
            {}, input_structure_required=False, request_height=1000,
            available_action_kinds=("tap_semantic", "swipe_element", "scroll", "home"),
        )["json_schema"]["schema"]["properties"]["decision"]["properties"]
        self.assertIn("point", schema)
        self.assertIn("start", schema)
        self.assertIn("end", schema)
        self.assertIn("effect", schema)
        for old in ("target", "element_id", "source_element_id", "destination_element_id", "tap_point"):
            self.assertNotIn(old, schema)

    def test_tap_uses_direct_point_without_scene_element(self) -> None:
        normalized = normalize_model_step_decision({
            "status": "action", "action": "tap", "point": [321, 654], "effect": "send_message",
        })
        action = bind_same_response_action(
            normalized, context=self.context, observation=self.observation,
            available_action_kinds={"tap_semantic"},
        )
        resolved = UniversalActionController().resolve_one(action, self.scene)
        self.assertEqual((0.321, 0.654), resolved.normalized_point)
        self.assertIsNone(resolved.target_element_id)
        self.assertEqual("send_message", action.params["effect_kind"])

    def test_swipe_uses_direct_start_and_end_without_element_binding(self) -> None:
        normalized = normalize_model_step_decision({
            "status": "action", "action": "swipe", "start": [500, 800], "end": [500, 200],
        })
        action = bind_same_response_action(
            normalized, context=self.context, observation=self.observation,
            available_action_kinds={"swipe_element"},
        )
        resolved = UniversalActionController().resolve_one(action, self.scene)
        self.assertEqual((0.5, 0.8), resolved.normalized_point)
        self.assertEqual((0.5, 0.2), resolved.normalized_end_point)
        self.assertIsNone(resolved.target_element_id)

    def test_old_element_fields_are_rejected(self) -> None:
        with self.assertRaisesRegex(CanonicalActionProtocolError, "协议外字段"):
            normalize_model_step_decision({
                "status": "action", "action": "tap", "point": [1, 2],
                "element_id": "old-target",
            })

    def test_swipe_without_complete_coordinates_is_rejected(self) -> None:
        with self.assertRaisesRegex(CanonicalActionProtocolError, "起点和终点"):
            normalize_model_step_decision({
                "status": "action", "action": "swipe", "start": [1, 2],
            })


if __name__ == "__main__":
    unittest.main()

