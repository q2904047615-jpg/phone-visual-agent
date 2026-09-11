from __future__ import annotations
from agent.infrastructure.generic_scene_observer import SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION
from agent.infrastructure.generic_scene_observer import SingleStepGenericSceneObserver
from agent.domain.ui_scene import UI_SCENE_PROTOCOL_VERSION
from agent.domain.vision_model import VisionAgentError
from agent.infrastructure.dashscope_vision_provider import _extract_json_object
from agent.infrastructure.generic_scene_observer import _parse_single_step_observation_envelope
from agent.infrastructure.generic_scene_observer import _single_step_response_format
import json
import unittest
from test_support.generic_scene_observer import (
    SequenceProvider,
    _BaseSingleStepGenericSceneObserverTests,
    _BaseStructuredDecisionContractTests,
    audited_text_input_scene,
    default_model_decision,
    scene_payload,
    stable_frames,
)


class StructuredDecisionContractTests(_BaseStructuredDecisionContractTests):
    def test_pending_transition_schema_forbids_finish_and_requires_scoped_action(self) -> None:
        response_format = _single_step_response_format(self._context("pending"),
            input_structure_required=True, request_height=1778,
            available_action_kinds=("clear_verified_text", "input_verified_text"))
        wrapper = response_format["json_schema"]
        decision = wrapper["schema"]["properties"]["decision"]

        self.assertIs(wrapper["strict"], True)
        self.assertNotIn("oneOf", wrapper["schema"])
        self.assertEqual(["action", "finish"], decision["properties"]["status"]["enum"])
        self.assertEqual(set(decision["properties"]), set(decision["required"]))
        self.assertEqual(["clear_verified_text", "input_verified_text", None],
            decision["properties"]["action"]["enum"])
        self.assertEqual([1778], wrapper["schema"]["properties"]["coordinate_space"]["properties"][
            "height"]["enum"])

    def test_executed_transition_schema_restores_same_frame_finish_choice(self) -> None:
        response_format = _single_step_response_format(self._context("executed"),
            input_structure_required=True, request_height=1778,
            available_action_kinds=("input_verified_text",))
        decision = response_format["json_schema"]["schema"]["properties"]["decision"]

        self.assertEqual(["action", "finish"], decision["properties"]["status"]["enum"])
        self.assertNotIn("evidence_refs", decision["properties"])
        self.assertEqual(["string", "null"], decision["properties"]["action"]["type"])


class SingleStepGenericSceneObserverTests(_BaseSingleStepGenericSceneObserverTests):
    def test_single_step_accepts_exactly_one_object_wrapped_in_array(self) -> None:
        payload = {
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "axis_grid", "width": 1000, "height": 960},
            "scene": scene_payload(),
            "input_structure": None,
            "decision": {"status": "finish", "evidence_refs": ["scene.summary"]},
        }
        provider = SequenceProvider([json.dumps([payload], ensure_ascii=False)])

        observed, decision = SingleStepGenericSceneObserver(provider).observe_with_decision(
            frames=stable_frames(), goal_context={"objective": "确认当前页面"},
            device_id="device-local-01")

        self.assertEqual("app_home", observed.screen_id)
        self.assertEqual("finish", decision["status"])
        self.assertEqual(1, provider.calls)

    def test_single_step_rejects_ambiguous_or_nonobject_arrays_without_retry(self) -> None:
        payload = {
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "axis_grid", "width": 1000, "height": 960},
            "scene": scene_payload(),
            "input_structure": None,
            "decision": {"status": "finish", "evidence_refs": ["scene.summary"]},
        }
        cases = ([], [payload, payload], ["not-an-object"])
        for value in cases:
            with self.subTest(value_type=type(value[0]).__name__ if value else "empty",
                item_count=len(value)):
                provider = SequenceProvider([json.dumps(value, ensure_ascii=False)])
                observer = SingleStepGenericSceneObserver(provider)

                with self.assertRaisesRegex(VisionAgentError, "恰好包含一个 JSON 对象"):
                    observer.observe(frames=stable_frames())

                self.assertEqual(1, provider.calls)
                self.assertFalse(observer.last_diagnostics["remote_retry_used"])

    def test_single_step_singleton_array_keeps_duplicate_key_rejection(self) -> None:
        raw = ('[{"protocol_version":"' + SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION
            + '","coordinate_space":{"kind":"axis_grid","width":1000,"height":960},'
            + '"scene":{},"decision":{"status":"finish"},'
            + '"decision":{"status":"finish"}}]')
        provider = SequenceProvider([raw])

        with self.assertRaisesRegex(VisionAgentError, "重复字段"):
            SingleStepGenericSceneObserver(provider).observe(frames=stable_frames())

        self.assertEqual(1, provider.calls)

    def test_non_single_step_json_extraction_remains_object_only(self) -> None:
        with self.assertRaisesRegex(VisionAgentError, "必须是 JSON 对象"):
            _extract_json_object('[{"value":1}]')

    def test_element_swipe_axis_grid_points_normalize_with_current_image_height(self) -> None:
        payload = {
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "axis_grid", "width": 1000, "height": 960},
            "scene": scene_payload(),
            "input_structure": None,
            "decision": {"status": "action", "action": "swipe_element",
                "element_id": "e1", "start": [180, 680], "end": [500, 680]},
        }
        observed, decision = SingleStepGenericSceneObserver(
            SequenceProvider([payload])
        ).observe_with_decision(
            frames=stable_frames(), goal_context={"objective": "移走当前目标"},
            device_id="device-local-01", available_action_kinds={"swipe_element"},
        )

        self.assertIsNotNone(observed.get_element("e1"))
        self.assertEqual([180, 708], decision["start"])
        self.assertEqual([500, 708], decision["end"])

    def test_obsolete_blocked_decision_is_rejected_by_wire_contract(self) -> None:
        payload = {
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {
                "kind": "normalized_1000",
                "width": 1000,
                "height": 1000,
            },
            "scene": scene_payload(),
            "input_structure": None,
            "decision": {
                "status": "blocked",
                "action": None,
                "element_id": None,
                "source_element_id": None,
                "destination_element_id": None,
                "direction": None,
                "evidence_refs": [],
                "confidence": 0.9,
                "reason": "legacy model veto",
            },
        }
        provider = SequenceProvider([payload])

        with self.assertRaisesRegex(VisionAgentError, "只允许action或finish"):
            SingleStepGenericSceneObserver(provider).observe(
                frames=stable_frames(),
                goal_context={"objective": "返回手机主屏幕"},
                device_id="device-local-01",
                available_action_kinds={"home"},
            )

    def test_missing_fixed_outer_protocol_version_is_filled_once(self) -> None:
        payload = {
            "coordinate_space": {
                "kind": "normalized_1000",
                "width": 1000,
                "height": 1000,
            },
            "scene": scene_payload(),
            "input_structure": None,
            "decision": {"status": "finish", "evidence_refs": ["scene.summary"]},
        }
        provider = SequenceProvider([payload])

        observed = SingleStepGenericSceneObserver(provider).observe(
            frames=stable_frames(),
            goal_context={"objective": "确认当前页面"},
            device_id="device-local-01",
        )

        self.assertEqual("app_home", observed.screen_id)
        self.assertEqual(1, provider.calls)

    def test_explicit_conflicting_outer_protocol_version_is_rejected(self) -> None:
        payload = {
            "protocol_version": "conflicting-version",
            "coordinate_space": {
                "kind": "normalized_1000",
                "width": 1000,
                "height": 1000,
            },
            "scene": scene_payload(),
            "input_structure": None,
            "decision": {"status": "finish", "evidence_refs": ["scene.summary"]},
        }
        provider = SequenceProvider([payload])

        with self.assertRaisesRegex(VisionAgentError, "协议版本不匹配"):
            SingleStepGenericSceneObserver(provider).observe(
                frames=stable_frames(),
                goal_context={"objective": "确认当前页面"},
                device_id="device-local-01",
            )

        self.assertEqual(1, provider.calls)

    def test_nested_action_injection_inside_harmless_outer_metadata_is_rejected(self) -> None:
        payload = {
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "normalized_1000", "width": 1000, "height": 1000},
            "scene": scene_payload(),
            "input_structure": None,
            "decision": {"status": "finish", "evidence_refs": ["scene.summary"]},
            "metadata": {"plan": ["click once", "click again"]},
        }
        provider = SequenceProvider([payload])

        with self.assertRaisesRegex(VisionAgentError, "动作或计划"):
            SingleStepGenericSceneObserver(provider).observe(
                frames=stable_frames(), goal_context={"objective": "确认当前页面"},
                device_id="device-local-01")

        self.assertEqual(1, provider.calls)

    def test_missing_optional_system_ui_and_camera_use_local_frame_facts(self) -> None:
        compact_scene = scene_payload()
        compact_scene.pop("system_ui")
        compact_scene.pop("camera_alignment")
        provider = SequenceProvider(
            [
                {
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    "coordinate_space": {
                        "kind": "normalized_1000",
                        "width": 1000,
                        "height": 1000,
                    },
                    "scene": compact_scene,
                    "input_structure": None,
                    "decision": {
                        "status": "finish",
                        "evidence_refs": ["scene.summary"],
                    },
                }
            ]
        )

        observed = SingleStepGenericSceneObserver(provider).observe(
            frames=stable_frames(),
            goal_context={"objective": "确认当前页面"},
            device_id="device-local-01",
        )

        self.assertEqual("unknown", observed.system_ui.navigation_bar_visible)
        self.assertEqual("portrait", observed.camera_alignment.camera_layout_orientation)
        self.assertEqual("unknown", observed.camera_alignment.phone_content_rotation)

    def test_duplicate_finish_evidence_is_normalized_without_a_second_veto(self) -> None:
        provider = SequenceProvider(
            [
                {
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    "coordinate_space": {
                        "kind": "normalized_1000",
                        "width": 1000,
                        "height": 1000,
                    },
                    "scene": scene_payload(),
                    "input_structure": None,
                    "decision": {
                        "status": "finish",
                        "evidence_refs": ["scene.summary", "scene.summary"],
                    },
                }
            ]
        )
        observer = SingleStepGenericSceneObserver(provider)

        observed, model_decision = observer.observe_with_decision(
            frames=stable_frames(),
            goal_context={"objective": "确认当前页面"},
            device_id="device-local-01",
        )

        self.assertNotIn("evidence_refs", model_decision)
        self.assertEqual([None], provider.max_tokens_seen)
        self.assertEqual("json_schema", provider.call_options["response_format"]["type"])
        schema = provider.call_options["response_format"]["json_schema"]["schema"]
        self.assertNotIn("oneOf", schema)
        self.assertEqual(["action", "finish"],
            schema["properties"]["decision"]["properties"]["status"]["enum"])

    def test_strict_direct_target_rejects_bounds_geometry(self) -> None:
        scene = scene_payload()
        scene["elements"] = []
        provider = SequenceProvider([{
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "normalized_1000", "width": 1000, "height": 1000},
            "scene": scene,
            "decision": {"status": "action", "action": "tap_semantic",
                "target": {"element_id": "selected-target", "role": "button", "meaning": "open_target",
                    "label": "打开", "evidence": ["打开"], "bounds": [200, 400, 500, 500]},
                "tap_point": [500, 500]},
        }])

        with self.assertRaisesRegex(VisionAgentError, "decision.target不得携带几何"):
            SingleStepGenericSceneObserver(provider).observe(
                frames=stable_frames(), goal_context={"objective": "打开目标"},
                device_id="device-local-01")

    def test_direct_point_branch_discards_duplicate_scene_geometry(self) -> None:
        scene = scene_payload()
        selected = {
            "element_id": "selected-target",
            "role": "button",
            "meaning": "open_target",
            "label": "打开",
            "bounds": [200, 400, 500, 500],
            "states": {"goal_relevant": True, "visible": True, "enabled": True},
        }
        scene["elements"] = [selected, {**selected, "bounds": [550, 400, 850, 500]}]
        provider = SequenceProvider([{
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "normalized_1000", "width": 1000, "height": 1000},
            "scene": scene,
            "decision": {"status": "action", "action": "tap_semantic",
                "target": {"element_id": "selected-target", "role": "button", "meaning": "open_target",
                    "label": "打开", "evidence": ["打开"]}, "tap_point": [500, 500]},
        }])

        observer = SingleStepGenericSceneObserver(provider)
        scene, decision = observer.observe_with_decision(
            frames=stable_frames(), goal_context={"objective": "打开目标"},
            device_id="device-local-01")
        self.assertEqual((), scene.elements)
        self.assertEqual("selected-target", decision['target']['element_id'])
        raw = json.loads(observer.last_raw_response)
        self.assertEqual([500, round(500 * 1000 / raw['coordinate_space']['height'])], decision['tap_point'])
        self.assertEqual(2, len(json.loads(observer.last_raw_response)['scene']['elements']))

    def test_invalid_unselected_goal_hint_does_not_veto_selected_scene_element(self) -> None:
        scene = scene_payload()
        scene["elements"][0].update(
            element_id="selected-target", meaning="open_target",
            states={"goal_relevant": True, "visible": True, "enabled": True})
        scene["elements"].append({
            "element_id": "bad-optional-hint",
            "role": "button",
            "meaning": "other_target",
            "bounds": [800, 300, 700, 400],
            "states": {"goal_relevant": True},
        })
        provider = SequenceProvider([{
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "normalized_1000", "width": 1000, "height": 1000},
            "scene": scene,
            "decision": {"status": "action", "action": "swipe_element",
                "element_id": "selected-target", "start": [400, 450], "end": [200, 450]},
        }])

        observed = SingleStepGenericSceneObserver(provider).observe(
            frames=stable_frames(), goal_context={"objective": "打开目标"},
            device_id="device-local-01")

        self.assertEqual(["selected-target"], [item.element_id for item in observed.elements])

    def test_malformed_optional_states_do_not_veto_selected_scene_element(self) -> None:
        scene = scene_payload()
        scene["elements"] = [{
            "element_id": "selected-target",
            "role": "button",
            "meaning": "open_target",
            "label": "打开",
            "bounds": [200, 400, 500, 500],
            "confidence": 0.99,
            "states": {"keyboard_layout": "qwerty"},
            "evidence": ["当前截图中完整可见的打开按钮"],
        }]
        provider = SequenceProvider([{
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "normalized_1000", "width": 1000, "height": 1000},
            "scene": scene,
            "decision": {"status": "action", "action": "swipe_element",
                "element_id": "selected-target", "start": [400, 450], "end": [200, 450]},
        }])

        observed = SingleStepGenericSceneObserver(provider).observe(
            frames=stable_frames(), goal_context={"objective": "打开目标"},
            device_id="device-local-01")

        selected = observed.get_element("selected-target")
        self.assertEqual("open_target", selected.meaning)
        self.assertEqual({"keyboard_layout": "qwerty"}, selected.states)
        self.assertEqual(1, provider.calls)

    def test_invalid_unselected_element_semantics_are_dropped_without_veto(self) -> None:
        scene = scene_payload()
        scene["elements"][0].update(
            element_id="selected-target", meaning="open_target",
            states={"goal_relevant": True, "visible": True, "enabled": True})
        scene["elements"].append({
            "element_id": "bad-optional-semantic-hint",
            "role": "future_widget",
            "meaning": "unrelated_hint",
            "bounds": [100, 600, 400, 700],
            "confidence": 1.0,
            "states": {"goal_relevant": True},
        })
        provider = SequenceProvider([{
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "normalized_1000", "width": 1000, "height": 1000},
            "scene": scene,
            "decision": {"status": "action", "action": "swipe_element",
                "element_id": "selected-target", "start": [400, 450], "end": [200, 450]},
        }])

        observed = SingleStepGenericSceneObserver(provider).observe(
            frames=stable_frames(), goal_context={"objective": "打开目标"},
            device_id="device-local-01")

        self.assertEqual(["selected-target"], [item.element_id for item in observed.elements])
        self.assertEqual(1, provider.calls)

    def test_action_referenced_element_with_invalid_semantics_is_rejected(self) -> None:
        scene = scene_payload()
        scene["elements"] = [{
            "element_id": "selected-target",
            "role": "future_widget",
            "meaning": "open_target",
            "bounds": [100, 300, 500, 400],
            "confidence": 1.0,
            "states": {"goal_relevant": True},
        }]
        provider = SequenceProvider([{
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "normalized_1000", "width": 1000, "height": 1000},
            "scene": scene,
            "decision": {"status": "action", "action": "swipe_element",
                "element_id": "selected-target", "start": [400, 350], "end": [200, 350]},
        }])

        with self.assertRaisesRegex(VisionAgentError, "不支持的元素角色"):
            SingleStepGenericSceneObserver(provider).observe(
                frames=stable_frames(), goal_context={"objective": "打开目标"},
                device_id="device-local-01")

        self.assertEqual(1, provider.calls)

    def test_finish_optional_element_does_not_create_reference_gate(self) -> None:
        scene = scene_payload()
        scene["elements"] = [{
            "element_id": "bad-proof",
            "role": "text",
            "meaning": "",
            "label": "已完成",
            "bounds": [100, 300, 500, 400],
            "confidence": 1.0,
            "states": {"goal_relevant": True},
        }]
        provider = SequenceProvider([{
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "normalized_1000", "width": 1000, "height": 1000},
            "scene": scene,
            "decision": {"status": "finish", "evidence_refs": ["element:bad-proof"]},
        }])

        _, choice = SingleStepGenericSceneObserver(provider).observe_with_decision(
            frames=stable_frames(), goal_context={"objective": "确认目标完成"},
            device_id="device-local-01")
        self.assertEqual('finish', choice['status'])
        self.assertNotIn('evidence_refs', choice)

        self.assertEqual(1, provider.calls)

    def test_single_step_observer_rejects_retired_image_grid_contract(
        self,
    ) -> None:
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
        scene, audit = audited_text_input_scene([120, 870, 780, 915])
        retired = {
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {
                "kind": "image_grid",
                "width": 540,
                "height": 960,
            },
            "scene": scene,
            "input_structure": audit,
        }
        provider = SequenceProvider([json.dumps(retired, ensure_ascii=False)])

        with self.assertRaisesRegex(VisionAgentError, "coordinate_space"):
            SingleStepGenericSceneObserver(provider).observe(
                frames=stable_frames(),
                goal_context=context,
                device_id="device-local-01",
            )

        self.assertEqual(1, provider.calls)

    def test_direct_tap_point_normalizes_independently_from_coarse_bounds_across_layouts(self) -> None:
        cases = (
            ("com.tencent.mm", "chat_list", 1280, [50, 320, 950, 451], [500, 282]),
            ("com.android.settings", "settings_home", 960, [80, 384, 920, 480], [240, 400]),
        )
        expected_points = ([500, 220], [240, 417])
        for case, expected in zip(cases, expected_points):
            app_id, screen_id, height, bounds, tap_point = case
            payload = {
                "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                "coordinate_space": {"kind": "axis_grid", "width": 1000, "height": height},
                "scene": {
                    "protocol_version": UI_SCENE_PROTOCOL_VERSION,
                    "foreground_app_id": app_id,
                    "screen_id": screen_id,
                    "summary": "当前页面显示唯一目标条目",
                    "system_ui": {},
                    "camera_alignment": {},
                    "elements": [],
                    "overlays": [], "stable": True, "confidence": 0.9, "fingerprint": "wire-only",
                },
                "input_structure": None,
                "decision": {"status": "action", "action": "tap_semantic",
                    "target": {"element_id": "target-row", "role": "list_item",
                        "meaning": "open_target", "label": "目标条目", "evidence": ["目标条目"]},
                    "tap_point": tap_point},
            }
            with self.subTest(app_id=app_id, height=height):
                parsed = _parse_single_step_observation_envelope(json.dumps(payload, ensure_ascii=False),
                    input_structure_required=False, request_image_size=(720, height))
                self.assertEqual(expected, parsed["decision"]["tap_point"])
                center_y = round((bounds[1] + bounds[3]) * 500 / height)
                self.assertNotEqual(center_y, parsed["decision"]["tap_point"][1])

    def test_single_step_observer_rejects_retired_normalized_y_declaration(
        self,
    ) -> None:
        raw_bounds = [120, 870, 780, 915]
        scene, audit = audited_text_input_scene(raw_bounds)
        retired = {
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {
                "kind": "normalized_1000",
                "width": 1000,
                "height": 1000,
            },
            "scene": scene,
            "input_structure": audit,
        }
        provider = SequenceProvider([json.dumps(retired, ensure_ascii=False)])
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

        with self.assertRaisesRegex(VisionAgentError, "coordinate_space"):
            SingleStepGenericSceneObserver(provider).observe(
                frames=stable_frames(),
                goal_context=context,
                device_id="device-local-01",
            )

        self.assertEqual(1, provider.calls)

    def test_single_step_observer_rejects_unprovable_wire_coordinate_spaces(
        self,
    ) -> None:
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
        base_scene, audit = audited_text_input_scene([120, 870, 780, 915])
        cases = (
            (
                "retired_image_grid",
                {"kind": "image_grid", "width": 540, "height": 960},
                [120, 870, 780, 915],
                "coordinate_space",
            ),
            (
                "retired_normalized_grid",
                {"kind": "normalized_1000", "width": 1000, "height": 1000},
                [120, 870, 780, 915],
                "coordinate_space",
            ),
            (
                "wrong_axis_height",
                {"kind": "axis_grid", "width": 1000, "height": 1280},
                [120, 870, 780, 915],
                "coordinate_space",
            ),
        )
        for name, coordinate_space, bounds, error in cases:
            with self.subTest(name=name):
                scene = json.loads(json.dumps(base_scene, ensure_ascii=False))
                current_audit = json.loads(json.dumps(audit, ensure_ascii=False))
                scene["elements"][0]["bounds"] = list(bounds)
                current_audit["application_inputs"][0]["bounds"] = list(bounds)
                raw = {
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    "coordinate_space": coordinate_space,
                    "scene": scene,
                    "input_structure": current_audit,
                }
                provider = SequenceProvider([json.dumps(raw, ensure_ascii=False)])
                with self.assertRaisesRegex(VisionAgentError, error):
                    SingleStepGenericSceneObserver(provider).observe(
                        frames=stable_frames(),
                        goal_context=context,
                        device_id="device-local-01",
                    )
                self.assertEqual(1, provider.calls)

        provider = SequenceProvider(
            [
                json.dumps(
                    {
                        "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                        "scene": base_scene,
                        "input_structure": audit,
                    },
                    ensure_ascii=False,
                )
            ]
        )
        with self.assertRaisesRegex(VisionAgentError, "coordinate_space"):
            SingleStepGenericSceneObserver(provider).observe(
                frames=stable_frames(),
                goal_context=context,
                device_id="device-local-01",
            )
        self.assertEqual(1, provider.calls)

    def test_single_step_observer_does_not_retry_malformed_response(self) -> None:
        provider = SequenceProvider(["{not-json"])
        observer = SingleStepGenericSceneObserver(provider)

        with self.assertRaises(VisionAgentError):
            observer.observe(frames=stable_frames())

        self.assertEqual(provider.calls, 1)
        self.assertEqual(observer.last_diagnostics["model_calls"], 1)
        self.assertFalse(observer.last_diagnostics["remote_retry_used"])

    def test_single_step_observer_rejects_retired_flat_scene(self) -> None:
        provider = SequenceProvider([json.dumps(scene_payload(), ensure_ascii=False)])
        observer = SingleStepGenericSceneObserver(provider)

        with self.assertRaisesRegex(VisionAgentError, "coordinate_space"):
            observer.observe(frames=stable_frames())

        self.assertEqual(provider.calls, 1)
        self.assertEqual(observer.last_diagnostics["model_calls"], 1)

    def test_same_device_fingerprint_and_goal_still_call_qwen_for_each_observation(self) -> None:
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

        def current_scene_envelope() -> dict:
            return {
                "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                "coordinate_space": {
                    "kind": "axis_grid",
                    "width": 1000,
                    "height": 960,
                },
                "scene": scene_payload(),
                "input_structure": None,
                "decision": default_model_decision(),
            }

        provider = PlainSequenceProvider([current_scene_envelope(), current_scene_envelope()])
        observer = SingleStepGenericSceneObserver(provider)
        frames = stable_frames()
        context: dict = {}

        first = observer.observe(
            frames=frames,
            goal_context=context,
            device_id="device-live-a",
        )
        second = observer.observe(
            frames=frames,
            goal_context=context,
            device_id="device-live-a",
        )

        self.assertEqual(2, provider.calls)
        self.assertIsNot(first, second)
        self.assertEqual(1, observer.last_diagnostics["model_calls"])
        self.assertEqual("single_step_current_scene_observation", observer.last_diagnostics["strategy"])


if __name__ == "__main__":
    unittest.main()
