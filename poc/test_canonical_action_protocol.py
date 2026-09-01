from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
import unittest

from agent.domain.canonical_action_protocol import (
    CanonicalActionProtocolError,
    GenericStepProposal,
    bind_same_response_action,
)
from agent.domain.semantic_action import SemanticAction
from agent.domain.text_transport import TEXT_TRANSPORT_PROTOCOL, TextTransportProfile
from agent.domain.ui_scene import UIElement, UIScene


@dataclass(frozen=True)
class Observation:
    scene: UIScene

    def get_candidate(self, element_id: str) -> UIElement:
        return self.scene.get_element(element_id)


def element(element_id: str, *, role: str="button", meaning: str="target", label: str="目标",
    bounds: tuple[float, float, float, float]=(0.1, 0.2, 0.5, 0.35),
    states: dict | None=None) -> UIElement:
    return UIElement(element_id=element_id, role=role, meaning=meaning, label=label, bounds=bounds,
        confidence=0.42, states=dict(states or {"enabled": True, "fully_visible": True}), evidence=(label,))


def observation(*items: UIElement) -> Observation:
    return Observation(UIScene(app_id="sample.app", screen_id="sample", summary="当前测试画面",
        elements=tuple(items), stable=True, confidence=0.5, fingerprint="f" * 64))


def context(*, entities: dict | None=None, field_id: str="", operation: str="") -> SimpleNamespace:
    values = dict(entities or {})
    requested = None
    if operation == "input_verified_text":
        if field_id == "primary_input":
            requested = values.get("input_text")
        else:
            matches = [item for item in values.get("input_fields") or ()
                if isinstance(item, dict) and item.get("field_id") == field_id]
            requested = matches[0].get("text") if len(matches) == 1 else None
    return SimpleNamespace(revision=7, device_id="device-local-01", goal={"entities": values},
        current_subgoal={"input_field_id": field_id, "input_operation": operation},
        requested_input_text=requested)


def payload(action: str, **parts) -> dict:
    return {"status": "action", "action": action, **parts}


def ime_profile(*, device_id: str="device-local-01") -> TextTransportProfile:
    profile = TextTransportProfile(protocol_version=TEXT_TRANSPORT_PROTOCOL, profile_id="profile-local-01",
        device_id=device_id, pairing_id="pair-local-01", enabled=True,
        capabilities=("append_text", "clear_text"), ack_timeout_seconds=3.0)
    profile.validate()
    return profile


class DirectCanonicalBindingTests(unittest.TestCase):
    def test_qwen_element_action_binds_directly_without_second_catalog(self) -> None:
        current = element("target-button")
        action = bind_same_response_action(payload("tap_semantic", element_id=current.element_id),
            context=context(), observation=observation(current), available_action_kinds={"tap_semantic"})

        self.assertEqual("tap_semantic", action.action)
        self.assertEqual("target-button", action.params["element_id"])
        self.assertEqual((0.1, 0.2, 0.5, 0.35), current.bounds)
        self.assertFalse({"expected_effect", "expected_result", "formal_candidate_id", "formal_transition"}
            .intersection(action.params))

    def test_container_dialog_and_unknown_are_not_denied_only_by_role(self) -> None:
        for role in ("container", "dialog", "unknown"):
            with self.subTest(role=role):
                current = element(f"{role}-surface", role=role)
                action = bind_same_response_action(payload("tap_semantic", element_id=current.element_id),
                    context=context(), observation=observation(current), available_action_kinds={"tap_semantic"})
                self.assertEqual(role, action.params["role"])

    def test_current_element_identity_is_hard_but_optional_display_states_do_not_veto(self) -> None:
        enabled = element("enabled")
        with self.assertRaisesRegex(CanonicalActionProtocolError, "不存在或不唯一"):
            bind_same_response_action(payload("tap_semantic", element_id="old-frame-id"), context=context(),
                observation=observation(enabled), available_action_kinds={"tap_semantic"})

        for state in ({"visible": False}, {"enabled": False}, {"occluded": True},
            {"fully_visible": False}):
            with self.subTest(state=state):
                diagnostic = replace(enabled, states=state)
                action = bind_same_response_action(payload("tap_semantic", element_id="enabled"),
                    context=context(), observation=observation(diagnostic),
                    available_action_kinds={"tap_semantic"})
                self.assertEqual("enabled", action.params["element_id"])

    def test_device_action_kind_is_a_hard_check(self) -> None:
        current = element("target-button")
        with self.assertRaisesRegex(CanonicalActionProtocolError, "当前设备不支持"):
            bind_same_response_action(payload("tap_semantic", element_id=current.element_id), context=context(),
                observation=observation(current), available_action_kinds={"home"})

    def test_swipe_and_drag_bind_only_current_frame_data(self) -> None:
        source = element("source", bounds=(0.1, 0.1, 0.3, 0.25))
        destination = element("destination", bounds=(0.6, 0.6, 0.9, 0.8))
        current = observation(source, destination)

        swipe = bind_same_response_action(payload("swipe", direction="left", element_id="source"),
            context=context(), observation=current, available_action_kinds={"swipe"})
        self.assertEqual({"direction": "left", "element_id": "source", "target": "target", "role": "button",
            "label": "目标", "states": {"enabled": True, "fully_visible": True}}, swipe.params)

        drag = bind_same_response_action(payload("drag", source_element_id="source",
            destination_element_id="destination"), context=context(), observation=current,
            available_action_kinds={"drag"})
        self.assertEqual("source", drag.params["source_element_id"])
        self.assertEqual("destination", drag.params["destination_element_id"])
        self.assertFalse({"expected_effect", "formal_candidate_id", "formal_transition"}.intersection(drag.params))

    def test_launch_app_accepts_only_complete_trusted_registry_mapping(self) -> None:
        launch = {"launch_ref": "trusted:settings", "expected_app_id": "com.android.settings",
            "target_app_id": "com.android.settings", "target_app_name": "系统设置"}
        action = bind_same_response_action(payload("launch_app"), context=context(), observation=observation(),
            available_action_kinds={"launch_app"}, launch_target=launch)
        self.assertEqual("trusted:settings", action.params["launch_ref"])
        self.assertEqual("com.android.settings", action.params["expected_app_id"])
        self.assertNotIn("expected_effect", action.params)

        with self.assertRaisesRegex(CanonicalActionProtocolError, "唯一可信包名映射"):
            bind_same_response_action(payload("launch_app"), context=context(), observation=observation(),
                available_action_kinds={"launch_app"}, launch_target={"launch_ref": "trusted:settings"})

    def test_companion_input_binds_exact_typed_prior_fragment_and_expected(self) -> None:
        input_box = element("message-input", role="input", meaning="message_input", label="消息",
            states={"enabled": True, "fully_visible": True, "focused": True, "value": "aa",
                "input_field_id": "primary_input"})
        action = bind_same_response_action(payload("input_verified_text", element_id="message-input"),
            context=context(entities={"input_text": "aa你好"}, field_id="primary_input",
                operation="input_verified_text"), observation=observation(input_box),
            available_action_kinds={"input_verified_text"}, text_transport_profile=ime_profile())

        self.assertEqual("companion_ime", action.params["text_transport"])
        self.assertEqual("primary_input", action.params["input_field_id"])
        self.assertEqual("aa", action.params["prior_input_value"])
        self.assertEqual("你好", action.params["input_fragment"])
        self.assertEqual("aa你好", action.params["expected_input_value"])
        self.assertNotIn("expected_effect", action.params)

    def test_input_fields_bind_by_typed_current_subgoal_identity(self) -> None:
        input_box = element("second-input", role="input", meaning="form_input", label="姓氏",
            states={"focused": True, "value": "", "input_field_id": "last-name",
                "input_field_label": "姓氏", "enabled": True, "fully_visible": True})
        entities = {"input_fields": [{"field_id": "first-name", "field_label": "名字", "text": "Ada"},
            {"field_id": "last-name", "field_label": "姓氏", "text": "Lovelace"}]}
        action = bind_same_response_action(payload("input_verified_text", element_id="second-input"),
            context=context(entities=entities, field_id="last-name", operation="input_verified_text"),
            observation=observation(input_box),
            available_action_kinds={"input_verified_text"})
        self.assertEqual("Lovelace", action.params["text"])
        self.assertEqual("last-name", action.params["input_field_id"])

    def test_input_rejects_nonunique_typed_field_instead_of_guessing(self) -> None:
        input_box = element("unknown-input", role="input", meaning="form_input", label="输入",
            states={"focused": True, "value": "", "enabled": True, "fully_visible": True})
        entities = {"input_fields": [{"field_id": "first", "text": "one"},
            {"field_id": "second", "text": "two"}]}
        with self.assertRaisesRegex(CanonicalActionProtocolError, "typed input_field_id"):
            bind_same_response_action(payload("input_verified_text", element_id="unknown-input"),
                context=context(entities=entities, field_id="second", operation="input_verified_text"),
                observation=observation(input_box),
                available_action_kinds={"input_verified_text"})

    def test_clear_binds_exact_current_field_without_new_body_text(self) -> None:
        input_box = element("message-input", role="input", meaning="message_input", label="消息",
            states={"focused": True, "value": "old", "input_field_id": "message-body",
                "enabled": True, "fully_visible": True})
        action = bind_same_response_action(payload("clear_verified_text", element_id="message-input"),
            context=context(field_id="message-body", operation="clear_verified_text"),
            observation=observation(input_box), available_action_kinds={"clear_verified_text"},
            text_transport_profile=ime_profile())
        self.assertEqual("old", action.params["prior_input_value"])
        self.assertEqual("", action.params["expected_input_value"])
        self.assertNotIn("text", action.params)

    def test_proposal_keeps_only_action_or_finish_and_current_element(self) -> None:
        current = element("target-button")
        scene = observation(current).scene
        action = SemanticAction(node_id="qwen_visual_revision_7", action="tap_semantic",
            params={"element_id": "target-button", "target": "target", "role": "button", "label": "目标",
                "states": {"enabled": True, "fully_visible": True}})
        GenericStepProposal(status="action", action=action).validate(scene)
        GenericStepProposal(status="finish", reason="done").validate(scene)
        with self.assertRaisesRegex(CanonicalActionProtocolError, "action 或 finish"):
            GenericStepProposal(status="blocked").validate(scene)


if __name__ == "__main__":
    unittest.main()
