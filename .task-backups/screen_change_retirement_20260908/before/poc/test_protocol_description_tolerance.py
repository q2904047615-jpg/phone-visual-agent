"""Redundant descriptions never change execution; real parser/binder/controller."""
from copy import deepcopy
import json
import unittest

from agent.domain.canonical_action_protocol import (
    CanonicalActionProtocolError, normalize_model_step_decision,
)
from agent.domain.vision_model import VisionAgentError
from test_point_scene_projection import decision, observe, task_context, wire
from test_simple_input_completion_contract import saved_response


SYSTEM_ACTIONS = ("home", "back", "open_recent_apps", "reveal_system_navigation",
                  "wait_for_change", "launch_app")


class ProtocolDescriptionToleranceTests(unittest.TestCase):
    def test_all_system_actions_ignore_only_pure_description(self):
        for kind in SYSTEM_ACTIONS:
            baseline = decision(kind, app="便签" if kind == "launch_app" else None)
            expected = normalize_model_step_decision(baseline)
            for target in ({}, {"label": "返回主屏幕"}, {"role": "icon"},
                           {"role": None, "meaning": "", "label": None, "evidence": None},
                           {"role": "input", "meaning": "page_context", "element_id": "extra",
                            "label": "不参与选择的描述", "evidence": ["页面说明"]}):
                with self.subTest(kind=kind, target=target):
                    raw = {**baseline, "target": target}
                    snapshot = deepcopy(raw)
                    actual = normalize_model_step_decision(raw)
                    self.assertEqual(expected, actual)
                    self.assertEqual(snapshot, raw)
                    self.assertEqual(actual, normalize_model_step_decision(actual))

    def test_system_conflicting_execution_payload_still_rejects(self):
        for kind in SYSTEM_ACTIONS:
            baseline = decision(kind, app="便签" if kind == "launch_app" else None)
            for target in ({"bounds": [0, 0, 10, 10]}, {"tap_point": [10, 10]},
                           {"command": "example"}, {"actions": [{"action": "home"}]},
                           {"label": {"action": "home"}}, {"evidence": [{"shell": "example"}]},
                           {"element_id": ["extra"]}, "home", []):
                with self.subTest(kind=kind, target=target), self.assertRaises(CanonicalActionProtocolError):
                    normalize_model_step_decision({**baseline, "target": target})
            for key, value in {"tap_point": [100, 100], "start": [10, 10], "end": [20, 20],
                               "element_id": "another", "direction": "left", "text": "another"}.items():
                with self.subTest(kind=kind, key=key), self.assertRaises(CanonicalActionProtocolError):
                    normalize_model_step_decision({**baseline, "target": {"label": "说明"}, key: value})

    def test_system_description_does_not_create_input_audit_or_change_action(self):
        for kind in ("home", "back", "wait_for_change"):
            allowed = {kind}
            context = task_context("返回主屏幕" if kind == "home" else "继续当前操作")
            plain = wire(decision(kind))
            expected = observe(plain, context, allowed=allowed)[2]
            variant = deepcopy(plain)
            variant["decision"]["target"] = {"role": "input", "meaning": "ignored", "label": "说明"}
            _, result, actual, observer = observe(variant, context, allowed=allowed)
            self.assertEqual(expected, actual)
            self.assertEqual(kind, result.proposal.action.action)
            self.assertEqual(variant, json.loads(observer.last_raw_response))

    def test_all_text_actions_ignore_ids_in_real_observer_binding(self):
        for kind in ("input_verified_text", "clear_verified_text", "press_enter"):
            for foreground in ("app.notes", "app.chat"):
                payload = saved_response(1)
                payload["scene"].update(elements=[], foreground_app_id=foreground)
                payload["decision"].update(action=kind, element_id=None,
                    text="新内容" if kind == "input_verified_text" else None)
                payload["input_structure"]["application_inputs"][0].update(
                    text="" if kind == "input_verified_text" else "旧内容", focused=True,
                    multiline=True, preedit_text="")
                expected = observe(payload, task_context("编辑当前输入框"))[2]
                for identifier in ("model_extra_id", "missing", 42, {"diagnostic": "unused"}):
                    with self.subTest(kind=kind, app=foreground, identifier=identifier):
                        varied = deepcopy(payload)
                        varied["decision"]["element_id"] = identifier
                        _, result, actual, observer = observe(varied, task_context("编辑当前输入框"))
                        self.assertEqual(expected, actual)
                        self.assertEqual(kind, result.proposal.action.action)
                        self.assertEqual(varied, json.loads(observer.last_raw_response))
                        if kind == "press_enter":
                            self.assertEqual("\n", result.proposal.action.params["input_fragment"])
                            self.assertEqual("旧内容\n", result.proposal.action.params["expected_input_value"])

    def test_extra_id_never_overrides_focus_or_multiline(self):
        for focused, multiline in ((False, True), (None, True), (True, False), (True, None)):
            payload = saved_response(1)
            payload["scene"]["elements"] = []
            payload["decision"].update(action="press_enter", element_id="extra", text=None)
            payload["input_structure"]["application_inputs"][0].update(
                text="旧内容", focused=focused, multiline=multiline, preedit_text="")
            with self.subTest(focused=focused, multiline=multiline), self.assertRaises(VisionAgentError):
                observe(payload, task_context("当前输入框换行"))


if __name__ == "__main__":
    unittest.main()
