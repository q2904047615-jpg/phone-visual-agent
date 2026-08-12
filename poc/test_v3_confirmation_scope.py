from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from generic_action_adapter import GenericActionAdapterError
from generic_intent import GenericIntentDraft
from generic_step_planner import GenericStepProposal
from generic_supervised_runtime import GenericSupervisedSession
from semantic_executor import SemanticAction
from ui_scene import UIElement, UIScene


ROOT = Path(__file__).resolve().parent
DEEPSEEK_FIXTURE = json.loads(
    (ROOT / "frontend_contract_fixtures" / "deepseek_task_graph_v3.json").read_text(
        encoding="utf-8"
    )
)
QWEN_FIXTURE = json.loads(
    (ROOT / "frontend_contract_fixtures" / "qwen_visual_decision_v2.json").read_text(
        encoding="utf-8"
    )
)


class FakeController:
    def completion_evidence_after_action(self, *_args):
        return ()


class FakeResult:
    def __init__(self, before_scene, after_scene):
        self.before_scene = before_scene
        self.after_scene = after_scene
        self.resolved_action = object()
        self.physical_actions = 1

    def to_dict(self):
        return {"physical_actions": 1}


class RecordingAdapter:
    def __init__(self, *, fail=False):
        self.controller = FakeController()
        self.calls = 0
        self.fail = fail

    def execute(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise GenericActionAdapterError("offline executor failed", physical_actions=0)
        before = kwargs["planned_scene"]
        after = UIScene(
            app_id=before.app_id,
            screen_id="settings_home",
            summary="设置首页已显示",
            elements=before.elements,
            stable=True,
            confidence=0.96,
            fingerprint="after-one-action",
        )
        return FakeResult(before, after)


class NeverPlanner:
    def propose(self, *_args, **_kwargs):
        raise AssertionError("confirmation tests must not replan")


def make_session(*, decision_status="action", fail=False):
    decision = copy.deepcopy(QWEN_FIXTURE["decision"])
    decision["status"] = decision_status
    if decision_status != "action":
        decision["next_action"] = None
        decision["target_region"] = None
        decision["expected_result"] = {}
    action = SemanticAction(
        node_id="qwen_visual_revision_1",
        action="tap_semantic",
        params={
            "target": "open_settings",
            "label": "设置",
            "role": "icon",
            "states": {"goal_relevant": True},
            "element_id": "settings_icon",
            "expected_effect": {"scene_changed": True},
        },
    )
    scene = UIScene(
        app_id="launcher",
        screen_id="launcher_home",
        summary="桌面应用网格清晰可见",
        elements=(
            UIElement(
                element_id="settings_icon",
                role="icon",
                meaning="open_settings",
                label="设置",
                bounds=(0.68, 0.2, 0.86, 0.35),
                confidence=0.96,
                states={"goal_relevant": True},
            ),
        ),
        stable=True,
        confidence=0.95,
        fingerprint=decision["fingerprint"],
    )
    proposal = GenericStepProposal(
        status="action",
        action=action,
        reason="可信候选唯一且清晰。",
    )
    adapter = RecordingAdapter(fail=fail)
    run_dir = ROOT / "output" / "offline-v3-confirmation"
    run_dir.mkdir(parents=True, exist_ok=True)
    session = GenericSupervisedSession.start(
        session_id="session-v3-authority",
        device_id="phone-01",
        goal=GenericIntentDraft(
            understood=True,
            app_id="generic",
            app_name="目标应用",
            objective="完成当前通用目标",
            success_criteria={"visible": True},
        ),
        scene=scene,
        proposal=proposal,
        planner=NeverPlanner(),
        adapter=adapter,
        run_dir=run_dir,
    )
    session.bind_v3_confirmation_context(
        task_graph=copy.deepcopy(DEEPSEEK_FIXTURE["to_qwen_context"]),
        qwen_decision=decision,
    )
    return session, adapter


def confirmation(session):
    return {
        "session_id": session.session_id,
        "task_id": "task-map-001",
        "device_id": "phone-01",
        "revision": 1,
        "subgoal_id": "save_target",
        "risk_ids": ["save_place"],
    }


class V3ConfirmationScopeTests(unittest.TestCase):
    def test_correct_scope_executes_exactly_once_and_replay_is_rejected(self):
        session, adapter = make_session()
        result = session.confirm_v3(confirmed=True, confirmation=confirmation(session))
        self.assertEqual(result.physical_actions, 1)
        self.assertEqual(adapter.calls, 1)

        with self.assertRaisesRegex(GenericActionAdapterError, "已使用|已推进") as replay:
            session.confirm_v3(confirmed=True, confirmation=confirmation(session))
        self.assertEqual(replay.exception.physical_actions, 0)
        self.assertEqual(adapter.calls, 1)

    def test_each_scope_component_is_compared_against_current_authority(self):
        mutations = {
            "task": lambda value: value.update(task_id="another-task"),
            "device": lambda value: value.update(device_id="phone-02"),
            "revision": lambda value: value.update(revision=2),
            "subgoal": lambda value: value.update(subgoal_id="another-subgoal"),
            "risk_missing": lambda value: value.update(risk_ids=[]),
            "risk_extra": lambda value: value.update(risk_ids=["save_place", "extra"]),
            "risk_other": lambda value: value.update(risk_ids=["other"]),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                session, adapter = make_session()
                request_scope = confirmation(session)
                mutate(request_scope)
                with self.assertRaises(GenericActionAdapterError) as rejected:
                    session.confirm_v3(confirmed=True, confirmation=request_scope)
                self.assertEqual(rejected.exception.physical_actions, 0)
                self.assertEqual(adapter.calls, 0)

    def test_revision_change_after_authority_creation_is_toctou_rejected(self):
        session, adapter = make_session()
        session.task_graph["revision"] = 2
        session.task_graph["confirmation_gate"]["scope"]["revision"] = 2

        with self.assertRaisesRegex(GenericActionAdapterError, "变化|失效") as rejected:
            session.confirm_v3(confirmed=True, confirmation=confirmation(session))
        self.assertEqual(rejected.exception.physical_actions, 0)
        self.assertEqual(adapter.calls, 0)

    def test_any_authoritative_scope_change_after_grant_is_rejected(self):
        def change_task(session):
            session.task_graph["task_id"] = "task-changed"
            session.task_graph["confirmation_gate"]["scope"]["task_id"] = "task-changed"

        def change_device(session):
            session.task_graph["device_id"] = "phone-02"
            session.task_graph["confirmation_gate"]["scope"]["device_id"] = "phone-02"

        def change_subgoal(session):
            session.task_graph["current_subgoal"]["subgoal_id"] = "changed-subgoal"
            session.task_graph["confirmation_gate"]["scope"]["subgoal_id"] = "changed-subgoal"

        def change_risk(session):
            session.task_graph["current_subgoal"]["risk_action_ids"] = ["changed-risk"]
            session.task_graph["confirmation_gate"]["risk_ids"] = ["changed-risk"]

        mutations = {
            "task": change_task,
            "device": change_device,
            "subgoal": change_subgoal,
            "risk": change_risk,
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                session, adapter = make_session()
                mutate(session)
                with self.assertRaises(GenericActionAdapterError) as rejected:
                    session.confirm_v3(
                        confirmed=True,
                        confirmation=confirmation(session),
                    )
                self.assertEqual(rejected.exception.physical_actions, 0)
                self.assertEqual(adapter.calls, 0)

    def test_stale_observation_or_decision_is_rejected_without_execution(self):
        for stale in ("observation", "decision"):
            with self.subTest(stale=stale):
                session, adapter = make_session()
                if stale == "observation":
                    session.current_scene = UIScene(
                        app_id="launcher",
                        screen_id="changed",
                        summary="画面已变化",
                        elements=session.current_scene.elements,
                        stable=True,
                        confidence=0.95,
                        fingerprint="changed-fingerprint",
                    )
                else:
                    session.qwen_decision["observation_id"] = "obs_changed1234567890"
                with self.assertRaisesRegex(GenericActionAdapterError, "画面|决策|失效"):
                    session.confirm_v3(confirmed=True, confirmation=confirmation(session))
                self.assertEqual(adapter.calls, 0)

    def test_no_v3_authority_and_non_action_decisions_fail_closed(self):
        session, adapter = make_session()
        session.invalidate_v3_confirmation("test")
        with self.assertRaisesRegex(GenericActionAdapterError, "权威v3确认作用域|已使用"):
            session.confirm_v3(confirmed=True, confirmation=confirmation(session))
        self.assertEqual(adapter.calls, 0)

        for status in ("blocked", "finished"):
            with self.subTest(status=status):
                blocked_session, blocked_adapter = make_session(decision_status=status)
                with self.assertRaisesRegex(
                    GenericActionAdapterError,
                    "不可执行|没有等待确认|权威v3确认作用域",
                ):
                    blocked_session.confirm_v3(
                        confirmed=True,
                        confirmation=confirmation(blocked_session),
                    )
                self.assertEqual(blocked_adapter.calls, 0)

    def test_failed_execution_consumes_confirmation(self):
        session, adapter = make_session(fail=True)
        with self.assertRaisesRegex(GenericActionAdapterError, "offline executor failed"):
            session.confirm_v3(confirmed=True, confirmation=confirmation(session))
        self.assertEqual(adapter.calls, 1)
        with self.assertRaises(GenericActionAdapterError):
            session.confirm_v3(confirmed=True, confirmation=confirmation(session))
        self.assertEqual(adapter.calls, 1)

    def test_pause_and_termination_invalidate_authority(self):
        for transition in ("pause", "cancel"):
            with self.subTest(transition=transition):
                session, adapter = make_session()
                getattr(session, transition)()
                with self.assertRaisesRegex(GenericActionAdapterError, "已使用|已推进"):
                    session.confirm_v3(
                        confirmed=True,
                        confirmation=confirmation(session),
                    )
                self.assertEqual(adapter.calls, 0)

    def test_external_v3_and_blocked_or_finished_decisions_cannot_auto_run(self):
        session, adapter = make_session()
        with self.assertRaisesRegex(GenericActionAdapterError, "禁止自动推进"):
            session.run_safe_loop(confirmed=True, max_physical_actions=1)
        self.assertEqual(adapter.calls, 0)

        for status in ("blocked", "finished"):
            with self.subTest(status=status):
                blocked_session, blocked_adapter = make_session(decision_status=status)
                graph = blocked_session.task_graph
                graph["task_status"] = "running"
                graph["current_external_impact"] = "navigation_only"
                graph["current_subgoal"]["external_impact"] = "navigation_only"
                graph["current_subgoal"]["risk_action_ids"] = []
                graph["risk_actions"] = []
                graph["confirmation_gate"] = {
                    "required": False,
                    "state": "not_required",
                    "risk_ids": [],
                    "scope": {
                        "task_id": graph["task_id"],
                        "device_id": graph["device_id"],
                        "revision": graph["revision"],
                        "subgoal_id": graph["current_subgoal"]["subgoal_id"],
                    },
                    "external_state_action_allowed": False,
                }
                with self.assertRaisesRegex(GenericActionAdapterError, "没有自动执行入口"):
                    blocked_session.run_safe_loop(
                        confirmed=True,
                        max_physical_actions=1,
                    )
                self.assertEqual(blocked_adapter.calls, 0)


if __name__ == "__main__":
    unittest.main()
