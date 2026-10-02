from __future__ import annotations
from dataclasses import dataclass, replace
from types import SimpleNamespace
import unittest
from agent.domain.canonical_action_protocol import (
    CanonicalActionProtocolError,
    GenericStepProposal,
    bind_same_response_action,
    normalize_model_step_decision,
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


def context(**_ignored) -> SimpleNamespace:
    return SimpleNamespace(revision=7, device_id="device-local-01", exact_input_text=None)


def payload(action: str, **parts) -> dict:
    aliases = {"tap_semantic": "tap", "dismiss_overlay": "dismiss", "swipe_element": "swipe",
        "input_verified_text": "input", "clear_verified_text": "clear_input"}
    wire_action = aliases.get(action, action)
    if action in {"tap_semantic", "dismiss_overlay", "double_tap", "long_press"}:
        parts.pop("element_id", None)
        parts.pop("target_role", None)
        parts.pop("target_meaning", None)
        parts.pop("target_label", None)
        parts.pop("target", None)
        parts.setdefault("point", parts.pop("tap_point", [325, 275]))
    elif action in {"input_verified_text", "clear_verified_text"}:
        parts.pop("element_id", None)
        parts.pop("target", None)
    elif action == "swipe_element":
        parts.pop("element_id", None)
    elif action == "drag":
        parts.pop("source_element_id", None)
        parts.pop("destination_element_id", None)
        parts.setdefault("start", [100, 100])
        parts.setdefault("end", [200, 200])
    return {"status": "action", "action": wire_action, **parts}


def ime_profile(*, device_id: str="device-local-01") -> TextTransportProfile:
    profile = TextTransportProfile(protocol_version=TEXT_TRANSPORT_PROTOCOL, profile_id="profile-local-01",
        device_id=device_id, adb_serial="serial-local-01", enabled=True,
        capabilities=("append_text", "clear_text"), command_timeout_seconds=3.0)
    profile.validate()
    return profile


class DirectCanonicalBindingTests(unittest.TestCase):
    def test_qwen_element_action_binds_directly_without_second_catalog(self) -> None:
        current = element("target-button")
        action = bind_same_response_action(payload("tap_semantic", element_id=current.element_id,
            target_role=current.role, target_meaning=current.meaning, target_label=current.label,
            tap_point=[120, 180]),
            context=context(), observation=observation(current), available_action_kinds={"tap_semantic"})

        self.assertEqual("tap_semantic", action.action)
        self.assertNotIn("element_id", action.params)
        self.assertEqual((0.12, 0.18), action.params["tap_point"])
        self.assertEqual((0.1, 0.2, 0.5, 0.35), current.bounds)
        self.assertFalse({"expected_effect", "expected_result", "formal_candidate_id", "formal_transition"}
            .intersection(action.params))

    def test_direct_tap_point_is_required_and_does_not_inherit_coarse_bounds_center(self) -> None:
        current = element("adjacent-row", bounds=(0.05, 0.25, 0.95, 0.352))
        raw = payload("tap_semantic", element_id=current.element_id)
        raw.pop("point")
        with self.assertRaisesRegex(CanonicalActionProtocolError, "point"):
            normalize_model_step_decision(raw)

        action = bind_same_response_action(payload("tap_semantic", element_id=current.element_id,
            target_role=current.role, target_meaning=current.meaning, target_label=current.label,
            tap_point=[500, 220]), context=context(), observation=observation(current),
            available_action_kinds={"tap_semantic"})
        self.assertEqual((0.5, 0.22), action.params["tap_point"])
        self.assertNotEqual(current.center, action.params["tap_point"])

    def test_point_action_does_not_require_scene_element_role(self) -> None:
        for role in ("container", "dialog", "unknown"):
            with self.subTest(role=role):
                current = element(f"{role}-surface", role=role)
                action = bind_same_response_action(payload("tap_semantic"),
                    context=context(), observation=observation(current), available_action_kinds={"tap_semantic"})
                self.assertEqual((0.325, 0.275), action.params["tap_point"])

    def test_old_target_field_is_rejected_by_wire_contract(self) -> None:
        raw = payload("tap_semantic")
        raw["target"] = {"role": "button", "meaning": "legacy"}
        with self.assertRaisesRegex(CanonicalActionProtocolError, "协议外字段"):
            normalize_model_step_decision(raw)

    def test_device_action_kind_is_a_hard_check(self) -> None:
        current = element("target-button")
        with self.assertRaisesRegex(CanonicalActionProtocolError, "当前观察签发的动作集合不包含"):
            bind_same_response_action(payload("tap_semantic", element_id=current.element_id), context=context(),
                observation=observation(current), available_action_kinds={"home"})

    def test_scroll_element_swipe_and_drag_bind_only_current_frame_data(self) -> None:
        source = element("source", bounds=(0.1, 0.1, 0.3, 0.25))
        destination = element("destination", bounds=(0.6, 0.6, 0.9, 0.8))
        current = observation(source, destination)

        scroll = bind_same_response_action(payload("scroll", direction="left", element_id="source"),
            context=context(), observation=current, available_action_kinds={"scroll"})
        self.assertEqual({"direction": "left"}, scroll.params)

        element_swipe = bind_same_response_action(payload("swipe_element", element_id="source",
            start=[200, 180], end=[50, 180]), context=context(), observation=current,
            available_action_kinds={"swipe_element"})
        self.assertEqual((0.2, 0.18), element_swipe.params["start"])
        self.assertEqual((0.05, 0.18), element_swipe.params["end"])
        self.assertNotIn("element_id", element_swipe.params)

        drag = bind_same_response_action(payload("drag", start=[100, 100], end=[800, 800]),
            context=context(), observation=current, available_action_kinds={"drag"})
        self.assertEqual((0.1, 0.1), drag.params["start"])
        self.assertEqual((0.8, 0.8), drag.params["end"])
        self.assertFalse({"expected_effect", "formal_candidate_id", "formal_transition"}.intersection(drag.params))

    def test_element_swipe_requires_complete_in_frame_trajectory(self) -> None:
        current = observation(element("source", bounds=(0.1, 0.1, 0.5, 0.5)))
        with self.assertRaisesRegex(CanonicalActionProtocolError, "起点和终点"):
            normalize_model_step_decision(payload("swipe_element", start=[200, 200]))
        with self.assertRaisesRegex(CanonicalActionProtocolError, "坐标范围"):
            bind_same_response_action(payload("swipe_element",
                start=[200, 200], end=[1001, 200]), context=context(), observation=current,
                available_action_kinds={"swipe_element"})

    def test_launch_app_accepts_only_complete_trusted_registry_mapping(self) -> None:
        launch = {"launch_ref": "trusted:settings", "expected_app_id": "com.android.settings",
            "target_app_id": "com.android.settings", "target_app_name": "系统设置"}
        action = bind_same_response_action(payload("launch_app", app="系统设置"), context=context(), observation=observation(),
            available_action_kinds={"launch_app"}, launch_target=launch)
        self.assertEqual("trusted:settings", action.params["launch_ref"])
        self.assertEqual("com.android.settings", action.params["expected_app_id"])
        self.assertNotIn("expected_effect", action.params)

        with self.assertRaisesRegex(CanonicalActionProtocolError, "唯一可信包名映射"):
            bind_same_response_action(payload("launch_app"), context=context(), observation=observation(),
                available_action_kinds={"launch_app"}, launch_target={"launch_ref": "trusted:settings"})

    def test_adb_keyboard_input_binds_exact_typed_prior_fragment_and_expected(self) -> None:
        input_box = element("message-input", role="input", meaning="message_input", label="消息",
            states={"enabled": True, "fully_visible": True, "focused": True, "value": "aa",
                "input_field_id": "current_input"})
        action = bind_same_response_action(payload("input_verified_text", text="aa你好", element_id="message-input"),
            context=context(entities={"input_text": "aa你好"}, field_id="primary_input",
                operation="input_verified_text"), observation=observation(input_box),
            available_action_kinds={"input_verified_text"}, text_transport_profile=ime_profile())

        self.assertEqual("adb_keyboard", action.params["text_transport"])
        self.assertEqual("current_input", action.params["input_field_id"])
        self.assertEqual("aa", action.params["prior_input_value"])
        self.assertEqual("你好", action.params["input_fragment"])
        self.assertEqual("aa你好", action.params["expected_input_value"])
        self.assertNotIn("expected_effect", action.params)

    def test_old_input_target_description_is_rejected_by_wire_contract(self) -> None:
        raw = payload("input_verified_text", text="aa你好")
        raw["target"] = {"role": "input", "meaning": "message_input"}
        with self.assertRaisesRegex(CanonicalActionProtocolError, "协议外字段"):
            normalize_model_step_decision(raw)

    def test_adb_keyboard_input_rejects_unfocused_field_even_when_unique_and_visible(self) -> None:
        input_box = element("message-input", role="input", meaning="message_input", label="消息",
            states={"enabled": True, "fully_visible": True, "value": "",
                "soft_keyboard_visible": False, "input_field_id": "current_input"})

        with self.assertRaisesRegex(CanonicalActionProtocolError, "明确已聚焦"):
            bind_same_response_action(payload("input_verified_text", text="ADB测试？你好", element_id="message-input"),
                context=context(entities={"input_text": "ADB测试？你好"}, field_id="primary_input",
                    operation="input_verified_text"), observation=observation(input_box),
                available_action_kinds={"input_verified_text"}, text_transport_profile=ime_profile())

    def test_adb_keyboard_clear_rejects_old_text_without_current_focus(self) -> None:
        input_box = element("message-input", role="input", meaning="message_input", label="消息",
            states={"enabled": True, "fully_visible": True, "value": "aaazjie？你好",
                "soft_keyboard_visible": False, "input_field_id": "current_input"})

        with self.assertRaisesRegex(CanonicalActionProtocolError, "明确已聚焦"):
            bind_same_response_action(payload("clear_verified_text", element_id="message-input"),
                context=context(field_id="message-body", operation="clear_verified_text"),
                observation=observation(input_box), available_action_kinds={"clear_verified_text"},
                text_transport_profile=ime_profile())

    def test_input_binds_same_frame_selected_field(self) -> None:
        input_box = element("second-input", role="input", meaning="form_input", label="姓氏",
            states={"focused": True, "value": "", "input_field_id": "current_input",
                "input_field_label": "姓氏", "enabled": True, "fully_visible": True})
        entities = {"input_fields": [{"field_id": "first-name", "field_label": "名字", "text": "Ada"},
            {"field_id": "last-name", "field_label": "姓氏", "text": "Lovelace"}]}
        action = bind_same_response_action(payload("input_verified_text", text="Lovelace", element_id="second-input"),
            context=context(entities=entities, field_id="last-name", operation="input_verified_text"),
            observation=observation(input_box),
            available_action_kinds={"input_verified_text"}, text_transport_profile=ime_profile())
        self.assertEqual("Lovelace", action.params["text"])
        self.assertEqual("current_input", action.params["input_field_id"])

    def test_input_rejects_nonunique_typed_field_instead_of_guessing(self) -> None:
        input_box = element("unknown-input", role="input", meaning="form_input", label="输入",
            states={"focused": True, "value": "", "enabled": True, "fully_visible": True})
        entities = {"input_fields": [{"field_id": "first", "text": "one"},
            {"field_id": "second", "text": "two"}]}
        with self.assertRaisesRegex(CanonicalActionProtocolError, "没有唯一可执行字段"):
            bind_same_response_action(payload("input_verified_text", text="two", element_id="unknown-input"),
                context=context(entities=entities, field_id="second", operation="input_verified_text"),
                observation=observation(input_box),
                available_action_kinds={"input_verified_text"})

    def test_clear_binds_exact_current_field_without_new_body_text(self) -> None:
        input_box = element("message-input", role="input", meaning="message_input", label="消息",
            states={"focused": True, "value": "old", "input_field_id": "current_input",
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
