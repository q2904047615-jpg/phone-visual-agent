"""Offline contracts for the user's six explicit simplification choices."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import ast
import unittest
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr
from agent.domain.generic_goal import safe_goal_context
from agent.domain.qwen_task_context import QwenTaskContext
from agent.domain.text_input_utils import normalize_user_text
from agent.domain.ui_scene import UIElement
from agent.infrastructure.generic_scene_observer import _collect_audited_input_matches
from agent.domain.semantic_action import SemanticAction
from agent.domain.ui_scene import UIScene
from agent.domain.universal_action_controller import UniversalActionController, UniversalActionError
from agent.domain.canonical_action_protocol import normalize_model_step_decision


class UncappedTextTests(unittest.TestCase):
    def test_adb_binding_and_controller_preserve_literal_end_to_end(self):
        import test_canonical_action_protocol as fixture
        from agent.domain.canonical_action_protocol import bind_same_response_action
        for text in ("字" * 6000, "e\u0301\r\n\t🙂"):
            obs = fixture.observation(fixture.element("field", role="input", states={
                "input_field_id": "current_input", "focused": True, "value": ""}))
            bound = bind_same_response_action({"action": "input_verified_text", "text": text},
                context=fixture.context(entities={"input_text": text},
                    field_id="primary_input", operation="input_verified_text"),
                observation=obs, available_action_kinds={"input_verified_text"},
                text_transport_profile=fixture.ime_profile())
            result = UniversalActionController().resolve_one(bound, obs.scene)
            self.assertEqual(text, result.input_fragment)
            self.assertEqual(text, result.expected_input_value)

    def test_exact_unicode_is_preserved_without_nfc_or_size_rejection(self):
        for value in ("e\u0301\r\n\t🙂", "字" * 6000 + "\nEND", " " * 5):
            with self.subTest(size=len(value)):
                self.assertEqual(value, normalize_user_text(value, field_name="正文"))

    def test_plan_and_qwen_context_keep_long_literal_text(self):
        for value in ("字" * 6000, "e\u0301\r\n\tEND"):
            context = QwenTaskContext(task_id="literal", device_id="device-1", revision=1,
                raw_goal="输入正文", exact_input_text=value)
            context.validate()
            self.assertEqual(value, context.exact_input_text)

    def test_many_fields_recipients_and_long_labels_are_not_capacity_errors(self):
        fields = [{"field_id": f"field-{i}", "field_label": f"字段{i}" + "名" * 200,
            "text": "字" * 6000} for i in range(40)]
        entities = {"target_surface": "current_surface", "input_fields": fields,
            "recipients": [f"收件人{i}" + "名" * 200 for i in range(40)]}
        self.assertEqual(entities, safe_goal_context(entities))

    def test_context_preserves_lists_keys_strings_and_nesting(self):
        payload = {"label" * 30: ["内容" * 2000 for _ in range(60)]}
        for _ in range(8):
            payload = {"nested": payload}
        self.assertEqual(payload, safe_goal_context(deepcopy(payload)))

    def test_observed_long_draft_is_retained_for_clear(self):
        value = "草稿" * 3000
        matches = _collect_audited_input_matches([{"bounds": [100, 400, 900, 600], "text": value}])
        self.assertEqual([value], [item["text"] for item in matches])
        UIElement(element_id="field", role="input", meaning="body", label="",
            bounds=(.1, .4, .9, .6), confidence=.99, states={"value": value}).validate()

    def test_natural_web_request_has_no_literal_capacity_limit(self):
        from pydantic import StrictInt
        from agent.domain.execution_budget import DEFAULT_DEVICE_ACTION_BUDGET, DEFAULT_OBSERVATION_BUDGET
        source_path = Path(__file__).resolve().parents[2] / "agent" / "interfaces" / "http_models.py"
        source = source_path.read_text(encoding="utf-8")
        names = {"StrictAgentRequest", "GenericSupervisedStartRequest"}
        nodes = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name in names]
        namespace = dict(BaseModel=BaseModel, ConfigDict=ConfigDict, Field=Field,
            StrictBool=StrictBool, StrictStr=StrictStr, StrictInt=StrictInt,
            DEFAULT_DEVICE_ACTION_BUDGET=DEFAULT_DEVICE_ACTION_BUDGET,
            DEFAULT_OBSERVATION_BUDGET=DEFAULT_OBSERVATION_BUDGET, __name__="offline_request")
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "offline_request", "exec"), namespace)
        request = namespace["GenericSupervisedStartRequest"]
        request.model_rebuild(_types_namespace=namespace)
        value = "自然任务" * 2000
        self.assertEqual(value, request(text=value, device_id="device-1").text)


class UncappedGeometryAndObservationTests(unittest.TestCase):
    def scene(self):
        return UIScene(app_id="drawing", screen_id="canvas", summary="画布",
            elements=(UIElement(element_id="shape", role="image", meaning="shape", label="形状",
                bounds=(0, 0, 1, 1), confidence=.9, states={}),),
            stable=True, confidence=.9, fingerprint="fresh")

    def test_short_long_and_edge_gestures_are_allowed(self):
        controller = UniversalActionController()
        for start, end in (((.001, .5), (.002, .5)), ((0, .5), (1, .5)),
            ((.5, 0), (.5, .001)), ((.5, .001), (.5, 1))):
            with self.subTest(start=start, end=end):
                result = controller.resolve_one(SemanticAction("move", "swipe_element",
                    {"element_id": "shape", "start": start, "end": end}), self.scene())
                self.assertEqual(start, result.normalized_point)
                self.assertEqual(end, result.normalized_end_point)

    def test_invalid_geometry_is_still_rejected(self):
        for start, end in (((0, .5), (-.001, .5)), ((.5, .5), (.5, .5)),
            ((float("nan"), .5), (.6, .5)), ((.5, .5), (float("inf"), .5))):
            with self.subTest(start=start, end=end), self.assertRaises(UniversalActionError):
                UniversalActionController().resolve_one(SemanticAction("move", "swipe_element",
                    {"element_id": "shape", "start": start, "end": end}), self.scene())

    def test_target_evidence_and_label_are_not_truncated(self):
        label = "按钮" * 500
        evidence = [f"证据{i}" + "界面" * 300 for i in range(30)]
        result = normalize_model_step_decision({"status": "action", "action": "tap_semantic",
            "target": {"role": "button", "meaning": "activate", "label": label,
                "evidence": evidence}, "tap_point": [.5, .5]})
        self.assertEqual(label, result["target"]["label"])
        self.assertEqual(evidence, result["target"]["evidence"])

    def test_observation_contract_has_no_element_count_or_fixed_wording(self):
        from agent.infrastructure import generic_scene_observer as observer
        source = Path(observer.__file__).read_text(encoding="utf-8")
        self.assertNotIn("MAX_COMPACT_ELEMENTS", source)
        prompt = (Path(observer.__file__).parent / "prompts/compact_scene.txt").read_text(encoding="utf-8")
        for retired in ("{{MAX_ELEMENTS}}", "不超过40", "不超过60", "必须把目标条目及其之前所有",
            "meaning=page_title", "meaning=paged_viewport"):
            self.assertNotIn(retired, prompt)

    def test_point_identity_does_not_require_optional_evidence_text(self):
        for label, evidence in (("", []), ("按钮", ["可见按钮"])):
            result = UniversalActionController().resolve_one(SemanticAction("tap", "tap_semantic",
                {"element_id": "target", "target": "activate", "role": "button",
                    "label": label, "target_evidence": evidence, "tap_point": [.4, .5]}),
                replace(self.scene(), elements=()))
            self.assertEqual((.4, .5), result.normalized_point)


if __name__ == "__main__":
    unittest.main()
