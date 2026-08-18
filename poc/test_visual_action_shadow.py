import json
import unittest
import builtins
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from task_semantic_ir import EffectIntent, SemanticEntity, SurfaceRef, TaskSemanticIR
from ui_scene import UIElement, UIScene
from visual_action_shadow import (
    StateExpectation,
    TypedStateTransition,
    VisualActionShadowError,
    compile_visual_action_shadow,
    compile_visual_action_authority,
    compile_visual_action_shadow_safe,
    select_shadow_candidate,
)
from universal_agent_orchestrator import UniversalAgentOrchestrator


ALL_ACTIONS = {
    "tap_semantic",
    "dismiss_overlay",
    "swipe",
    "back",
    "home",
    "reveal_system_navigation",
    "input_verified_text",
    "clear_verified_text",
    "long_press",
    "drag",
    "wait_for_change",
}


def make_element(
    element_id="target",
    *,
    role="button",
    label="继续",
    meaning="literal_control",
    bounds=(0.1, 0.2, 0.4, 0.3),
    confidence=0.96,
    states=None,
    evidence=("模型原始自然语言证据不应进入 claim",),
):
    return UIElement(
        element_id=element_id,
        role=role,
        meaning=meaning,
        bounds=bounds,
        confidence=confidence,
        label=label,
        states={
            "visible": True,
            "enabled": True,
            "fully_visible": True,
            **(states or {}),
        },
        evidence=evidence,
    )


def make_scene(*elements, app_id="sample.app", screen_id="main"):
    return UIScene(
        app_id=app_id,
        screen_id=screen_id,
        summary="当前真实画面的非权威摘要",
        elements=tuple(elements),
        stable=True,
        confidence=0.95,
        fingerprint="f" * 64,
    )


def make_ir(
    *,
    role="target_ui_label",
    value="继续",
    raw_goal="找到继续并进入下一状态",
    surface=None,
    effect_kind="generic_effect",
    payload=False,
):
    entity = SemanticEntity(
        entity_id=f"entity_{role}",
        entity_type="text",
        role=role,
        value=value,
        authority="planner_context",
    )
    effect = EffectIntent(
        effect_id="effect_primary",
        kind=effect_kind,
        target_refs=() if payload else (entity.entity_id,),
        payload_refs=(entity.entity_id,) if payload else (),
        source_subgoal_ids=("subgoal_primary",),
    )
    return TaskSemanticIR(
        task_id="task-shadow-01",
        device_id="device-shadow-01",
        revision=1,
        raw_goal=raw_goal,
        surfaces=(surface or SurfaceRef("surface_current_goal", "current_surface"),),
        entities=(entity,),
        effects=(effect,),
    )


class VisualActionShadowTests(unittest.TestCase):
    def test_fixture_catalog_covers_required_change_samples(self):
        path = Path(__file__).parent / "fixtures" / "visual_action_shadow" / "cases.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["cases"]), 10)
        ids = {item["case_id"] for item in payload["cases"]}
        self.assertIn("launcher_exact_surface", ids)
        self.assertIn("focused_input_payload", ids)
        self.assertIn("recipient_exact_target", ids)
        self.assertIn("paraphrase_and_other_app", ids)
        self.assertIn("duplicate_literal", ids)
        self.assertIn("long_press_typed_exploration", ids)
        self.assertIn("drag_unique_typed_endpoints", ids)

    def test_report_is_deterministic_non_authoritative_and_ready(self):
        scene = make_scene(make_element())
        ir = make_ir()
        first = compile_visual_action_shadow(scene, ir, ALL_ACTIONS)
        second = compile_visual_action_shadow(scene, ir, reversed(sorted(ALL_ACTIONS)))
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.report_digest, second.report_digest)
        self.assertFalse(first.authoritative)
        self.assertFalse(first.execution_allowed)
        self.assertEqual(first.status, "ready")
        self.assertGreaterEqual(len(first.candidates), 2)
        self.assertLessEqual(len(first.candidates), 24)

    def test_goal_relevant_cannot_change_shadow_output(self):
        original = make_element(states={"goal_relevant": True})
        changed = make_element(states={"goal_relevant": False})
        ir = make_ir()
        one = compile_visual_action_shadow(make_scene(original), ir, ALL_ACTIONS)
        two = compile_visual_action_shadow(make_scene(changed), ir, ALL_ACTIONS)
        self.assertEqual(one.to_dict(), two.to_dict())

    def test_evidence_prose_is_replaced_by_digest(self):
        secret = "这段模型自然语言不得作为事实权威"
        report = compile_visual_action_shadow(
            make_scene(make_element(evidence=(secret,))),
            make_ir(),
            ALL_ACTIONS,
        )
        encoded = json.dumps(report.to_dict(), ensure_ascii=False)
        self.assertNotIn(secret, encoded)
        self.assertNotIn("evidence", encoded)
        self.assertRegex(report.claims[0].source_digest, r"^[0-9a-f]{64}$")

    def test_unique_literal_generates_typed_element_candidate(self):
        report = compile_visual_action_shadow(
            make_scene(make_element(label="项目群", role="list_item")),
            make_ir(role="recipient", value="项目群", effect_kind="send_message"),
            ALL_ACTIONS,
        )
        element_candidates = [
            item for item in report.candidates if item.action_kind == "tap_semantic"
        ]
        self.assertEqual(len(element_candidates), 1)
        candidate = element_candidates[0]
        self.assertEqual(candidate.effect_ref, "effect_primary")
        self.assertFalse(candidate.transition.exploratory)
        self.assertEqual(
            candidate.transition.expectations[0].predicate,
            "effect.applied",
        )
        relation_kinds = {
            item.relation
            for item in report.relations
            if item.relation_id in candidate.relation_ids
        }
        self.assertIn("exact_literal_match", relation_kinds)
        self.assertIn("binds_effect_target", relation_kinds)

    def test_launcher_icon_binds_target_surface_without_app_branch(self):
        surface = SurfaceRef(
            surface_id="surface_chat_tool",
            kind="app",
            app_id="chat.tool",
            app_name="聊天工具甲",
        )
        scene = make_scene(
            make_element(label="聊天工具甲", role="icon"),
            app_id="launcher",
            screen_id="home_screen",
        )
        report = compile_visual_action_shadow(
            scene,
            make_ir(value="无关文字", surface=surface),
            ALL_ACTIONS,
        )
        candidate = next(
            item for item in report.candidates if item.action_kind == "tap_semantic"
        )
        expectation = candidate.transition.expectations[0]
        self.assertEqual(expectation.predicate, "surface.active_ref")
        self.assertEqual(expectation.value, "surface_chat_tool")
        self.assertTrue(
            any(item.relation == "binds_surface" for item in report.relations)
        )

    def test_focused_input_binds_unique_typed_payload(self):
        input_element = make_element(
            role="input",
            label="消息",
            states={
                "focused": True,
                "value": "",
                "keyboard_layout": "qwerty",
                "keyboard_input_mode": "direct_latin",
            },
        )
        report = compile_visual_action_shadow(
            make_scene(input_element, app_id="another.product", screen_id="compose"),
            make_ir(
                role="input_text",
                value="任意文本 42",
                raw_goal="把给定内容写入当前输入区域",
                effect_kind="send_message",
                payload=True,
            ),
            ALL_ACTIONS,
        )
        candidate = next(
            item
            for item in report.candidates
            if item.action_kind == "input_verified_text"
        )
        expectation = candidate.transition.expectations[0]
        self.assertEqual(expectation.predicate, "element.state.value")
        self.assertEqual(expectation.value, "任意文本 42")
        self.assertFalse(candidate.transition.exploratory)

    def test_duplicate_literal_never_grants_effect_binding(self):
        scene = make_scene(
            make_element("first", label="小组", role="list_item"),
            make_element(
                "second",
                label="小组",
                role="list_item",
                bounds=(0.1, 0.4, 0.4, 0.5),
            ),
        )
        report = compile_visual_action_shadow(
            scene,
            make_ir(role="recipient", value="小组", effect_kind="send_message"),
            ALL_ACTIONS,
        )
        taps = [item for item in report.candidates if item.action_kind == "tap_semantic"]
        self.assertEqual(2, len(taps))
        self.assertTrue(all(not item.effect_ref for item in taps))
        self.assertIn("duplicate_exact_literal_binding", report.warnings)

    def test_low_confidence_or_incomplete_target_does_not_grant_action(self):
        for element in (
            make_element(confidence=0.4),
            make_element(states={"fully_visible": False}),
        ):
            with self.subTest(states=element.states, confidence=element.confidence):
                report = compile_visual_action_shadow(
                    make_scene(element), make_ir(), ALL_ACTIONS
                )
                self.assertFalse(
                    any(item.action_kind == "tap_semantic" for item in report.candidates)
                )

    def test_unrelated_goal_relevant_never_changes_element_candidate(self):
        first = compile_visual_action_shadow(
            make_scene(make_element(label="无关按钮", states={"goal_relevant": True})),
            make_ir(value="设置"),
            ALL_ACTIONS,
        )
        second = compile_visual_action_shadow(
            make_scene(make_element(label="无关按钮", states={"goal_relevant": False})),
            make_ir(value="设置"),
            ALL_ACTIONS,
        )
        self.assertEqual(first.to_dict(), second.to_dict())
        tap = next(item for item in first.candidates if item.action_kind == "tap_semantic")
        self.assertTrue(tap.transition.exploratory)

    def test_relational_scene_graph_has_surface_and_spatial_relations(self):
        report = compile_visual_action_shadow(
            make_scene(
                make_element("one", label="继续", bounds=(0.1, 0.1, 0.3, 0.2)),
                make_element("two", label="其他", bounds=(0.6, 0.1, 0.8, 0.2)),
            ),
            make_ir(),
            ALL_ACTIONS,
        )
        kinds = {item.relation for item in report.relations}
        self.assertIn("on_surface", kinds)
        self.assertTrue({"left_of", "right_of"}.intersection(kinds))
        self.assertIn("exact_literal_match", kinds)

    def test_regular_transition_cannot_use_scene_changed(self):
        transition = TypedStateTransition(
            transition_id="transition_invalid",
            precondition_claim_ids=("claim_valid",),
            expectations=(
                StateExpectation("surface_current", "scene.changed", "changed"),
            ),
            exploratory=False,
        )
        with self.assertRaisesRegex(VisualActionShadowError, "exploratory"):
            transition.validate()

    def test_exploratory_navigation_has_typed_weak_change(self):
        report = compile_visual_action_shadow(
            make_scene(make_element(states={"scrollable": True})), make_ir(), ALL_ACTIONS
        )
        swipe = next(item for item in report.candidates if item.action_kind == "swipe")
        self.assertTrue(swipe.transition.exploratory)
        self.assertEqual(
            swipe.transition.expectations[0].predicate,
            "surface.viewport",
        )
        ordinary = next(
            item for item in report.candidates if item.action_kind == "tap_semantic"
        )
        self.assertFalse(ordinary.transition.exploratory)
        self.assertNotEqual(
            ordinary.transition.expectations[0].predicate,
            "scene.changed",
        )

    def test_long_press_is_available_only_as_typed_exploration(self):
        report = compile_visual_action_shadow(
            make_scene(make_element()), make_ir(), ALL_ACTIONS
        )
        candidate = next(
            item for item in report.candidates if item.action_kind == "long_press"
        )
        self.assertTrue(candidate.transition.exploratory)
        self.assertEqual(
            candidate.transition.expectations[0].predicate,
            "element.state.interaction_result",
        )

    def test_drag_requires_unique_typed_source_and_destination(self):
        source = make_element(
            "source-control",
            label="卡片甲",
            role="list_item",
            bounds=(0.1, 0.2, 0.3, 0.3),
        )
        destination = make_element(
            "destination-control",
            label="区域乙",
            role="list_item",
            bounds=(0.6, 0.6, 0.9, 0.8),
        )
        entities = (
            SemanticEntity("entity_source", "opaque", "drag_source", "卡片甲"),
            SemanticEntity(
                "entity_destination", "opaque", "drag_destination", "区域乙"
            ),
        )
        ir = TaskSemanticIR(
            task_id="task-shadow-drag",
            device_id="device-shadow-01",
            revision=1,
            raw_goal="把卡片甲移动到区域乙",
            surfaces=(SurfaceRef("surface_current_goal", "current_surface"),),
            entities=entities,
            effects=(
                EffectIntent(
                    "effect_move",
                    "data_mutation",
                    target_refs=("entity_destination",),
                    payload_refs=("entity_source",),
                    source_subgoal_ids=("subgoal_move",),
                ),
            ),
        )
        report = compile_visual_action_shadow(
            make_scene(source, destination), ir, ALL_ACTIONS
        )
        candidate = next(item for item in report.candidates if item.action_kind == "drag")
        self.assertEqual(len(candidate.subject_refs), 2)
        self.assertEqual(
            candidate.transition.expectations[0].predicate,
            "element.state.location_relation",
        )
        self.assertFalse(candidate.transition.exploratory)

    def test_candidates_do_not_contain_coordinates(self):
        report = compile_visual_action_shadow(
            make_scene(make_element()), make_ir(), ALL_ACTIONS
        )
        encoded = json.dumps(
            [item.to_dict() for item in report.candidates], ensure_ascii=False
        )
        self.assertNotIn("bounds", encoded)
        self.assertNotIn('"x"', encoded)
        self.assertNotIn('"y"', encoded)

    def test_candidate_selection_only_returns_existing_candidate(self):
        report = compile_visual_action_shadow(
            make_scene(make_element()), make_ir(), ALL_ACTIONS
        )
        chosen = report.candidates[0]
        selection = select_shadow_candidate(
            report,
            report_digest=report.report_digest,
            candidate_id=chosen.candidate_id,
        )
        self.assertIs(selection.candidate, chosen)
        self.assertFalse(selection.authoritative)
        self.assertFalse(selection.execution_allowed)
        with self.assertRaises(FrozenInstanceError):
            selection.candidate.action_kind = "home"
        with self.assertRaisesRegex(VisualActionShadowError, "digest"):
            select_shadow_candidate(
                report,
                report_digest="0" * 64,
                candidate_id=chosen.candidate_id,
            )
        with self.assertRaisesRegex(VisualActionShadowError, "candidate_id"):
            select_shadow_candidate(
                report,
                report_digest=report.report_digest,
                candidate_id="candidate_missing",
            )

    def test_cross_app_and_paraphrase_use_same_candidate_shape(self):
        first = compile_visual_action_shadow(
            make_scene(make_element(label="搜索"), app_id="product.one", screen_id="a"),
            make_ir(value="搜索", raw_goal="找到搜索入口"),
            ALL_ACTIONS,
        )
        second = compile_visual_action_shadow(
            make_scene(make_element(label="搜索"), app_id="product.two", screen_id="b"),
            make_ir(value="搜索", raw_goal="进入可以查找内容的位置"),
            ALL_ACTIONS,
        )
        first_tap = next(item for item in first.candidates if item.action_kind == "tap_semantic")
        second_tap = next(item for item in second.candidates if item.action_kind == "tap_semantic")
        self.assertEqual(first_tap.action_kind, second_tap.action_kind)
        self.assertEqual(
            first_tap.transition.expectations[0].predicate,
            second_tap.transition.expectations[0].predicate,
        )

    def test_shadow_safe_boundary_never_grants_authority(self):
        report, error = compile_visual_action_shadow_safe(
            make_scene(make_element()), make_ir(), {"unknown_action"}
        )
        self.assertIsNone(report)
        self.assertIsNotNone(error)
        self.assertFalse(error["authoritative"])
        self.assertFalse(error["execution_allowed"])
        self.assertEqual(error["status"], "shadow_error")

    def test_single_typed_candidate_is_ready(self):
        report = compile_visual_action_shadow(
            make_scene(make_element(label="无匹配")),
            make_ir(value="其他"),
            {"wait_for_change"},
        )
        self.assertEqual(report.status, "ready")
        self.assertEqual(len(report.candidates), 1)

    def test_formal_authority_is_local_and_cannot_execute_by_itself(self):
        report = compile_visual_action_authority(
            make_scene(make_element(states={"goal_relevant": False})),
            make_ir(),
            ALL_ACTIONS,
        )
        self.assertTrue(report.authoritative)
        self.assertFalse(report.execution_allowed)
        self.assertTrue(any(item.action_kind == "tap_semantic" for item in report.candidates))

    def test_orchestrator_attaches_read_only_shadow_without_changing_decision(self):
        ir = make_ir()

        class Observer:
            def __init__(self):
                self.last_diagnostics = {"formal": "unchanged"}

            def decide(self, **kwargs):
                return {"formal_decision": "sentinel"}

        observer = Observer()
        planner = SimpleNamespace(
            last_semantic_shadow=SimpleNamespace(semantic_ir=ir)
        )
        orchestrator = UniversalAgentOrchestrator(
            deepseek_planner=planner,
            qwen_observer=observer,
            adapter_factory=lambda _device_id: None,
        )
        adapter = SimpleNamespace(
            supported_action_kinds=lambda: ALL_ACTIONS,
        )
        result = orchestrator._decide_next_action(
            SimpleNamespace(step_number=1, adapter=adapter),
            frames=[],
            task_context={
                "task_id": ir.task_id,
                "device_id": ir.device_id,
                "revision": ir.revision,
            },
            trusted_observation=SimpleNamespace(
                scene=make_scene(make_element())
            ),
        )
        self.assertEqual(result, {"formal_decision": "sentinel"})
        self.assertEqual(observer.last_diagnostics["formal"], "unchanged")
        shadow = observer.last_diagnostics["visual_action_shadow"]
        self.assertTrue(shadow["authoritative"])
        self.assertFalse(shadow["execution_allowed"])
        self.assertGreaterEqual(shadow["candidate_count"], 2)

    def test_shadow_scope_mismatch_never_changes_formal_decision(self):
        ir = make_ir()

        class Observer:
            def __init__(self):
                self.last_diagnostics = {}

            def decide(self, **kwargs):
                return "formal-result"

        observer = Observer()
        orchestrator = UniversalAgentOrchestrator(
            deepseek_planner=SimpleNamespace(
                last_semantic_shadow=SimpleNamespace(semantic_ir=ir)
            ),
            qwen_observer=observer,
            adapter_factory=lambda _device_id: None,
        )
        result = orchestrator._decide_next_action(
            SimpleNamespace(
                step_number=1,
                adapter=SimpleNamespace(
                    supported_action_kinds=lambda: ALL_ACTIONS
                ),
            ),
            frames=[],
            task_context={
                "task_id": "different-task",
                "device_id": ir.device_id,
                "revision": ir.revision,
            },
            trusted_observation=SimpleNamespace(
                scene=make_scene(make_element())
            ),
        )
        self.assertEqual(result, "formal-result")
        self.assertIsNone(orchestrator.last_visual_action_shadow)
        self.assertEqual(
            observer.last_diagnostics["visual_action_shadow"]["error_type"],
            "ShadowScopeMismatch",
        )

    def test_shadow_import_failure_cannot_break_formal_decision(self):
        ir = make_ir()

        class Observer:
            def __init__(self):
                self.last_diagnostics = {}

            def decide(self, **kwargs):
                return "formal-still-wins"

        observer = Observer()
        orchestrator = UniversalAgentOrchestrator(
            deepseek_planner=SimpleNamespace(
                last_semantic_shadow=SimpleNamespace(semantic_ir=ir)
            ),
            qwen_observer=observer,
            adapter_factory=lambda _device_id: None,
        )
        real_import = builtins.__import__

        def rejecting_import(name, *args, **kwargs):
            if name == "visual_action_shadow":
                raise ImportError("shadow module unavailable")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=rejecting_import):
            result = orchestrator._decide_next_action(
                SimpleNamespace(
                    step_number=1,
                    adapter=SimpleNamespace(
                        supported_action_kinds=lambda: ALL_ACTIONS
                    ),
                ),
                frames=[],
                task_context={
                    "task_id": ir.task_id,
                    "device_id": ir.device_id,
                    "revision": ir.revision,
                },
                trusted_observation=SimpleNamespace(
                    scene=make_scene(make_element())
                ),
            )
        self.assertEqual(result, "formal-still-wins")
        diagnostic = observer.last_diagnostics["visual_action_shadow"]
        self.assertEqual(diagnostic["error_type"], "ImportError")
        self.assertFalse(diagnostic["authoritative"])
        self.assertFalse(diagnostic["execution_allowed"])


if __name__ == "__main__":
    unittest.main()
