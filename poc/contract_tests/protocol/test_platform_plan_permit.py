"""Runtime plan commit, Verifier plan conditions and one-shot execution permits."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from agent.domain.task_plan import (
    ExecutionPermit, GOAL_FALLBACK_SOURCE, MODEL_PROPOSAL_SOURCE, TaskPlan, TaskPlanError,
)
from test_single_visual_loop import LoopHarness, decision, scene


def proposing(base: dict, conditions) -> dict:
    return {**base, 'success_conditions': list(conditions)}


def finishing(results) -> dict:
    return {**decision(), 'condition_results': list(results)}


MET = [{'id': 'c1', 'status': 'met', 'evidence': '页面标题为蓝牙'},
       {'id': 'c2', 'status': 'met', 'evidence': '蓝牙开关为开启'}]
CONDITIONS = ('当前页面是蓝牙设置', '蓝牙开关显示为开启')


class RuntimePlanCommitTests(LoopHarness):
    def test_first_decision_commits_model_proposed_conditions_as_plan_v1(self):
        rows = [(scene(0), proposing(decision('tap_semantic'), CONDITIONS)), (scene(1), finishing(MET))]
        loop, session, adapter = self.start(rows, goal='打开蓝牙')
        plan = session.task_plan
        self.assertEqual((1, MODEL_PROPOSAL_SOURCE), (plan.version, plan.source))
        self.assertEqual(('c1', 'c2'), plan.condition_ids)
        self.assertEqual(CONDITIONS, tuple(item.description for item in plan.success_conditions))
        self.assertEqual(plan.to_dict(), session.snapshot()['task_plan'])
        evidence = json.loads(Path(session.run_dir, 'task_plan_v1.json').read_text(encoding='utf-8'))
        self.assertEqual(plan.plan_id, evidence['plan_id'])
        loop.run_autonomous_safe_loop(session)
        self.assertEqual('succeeded', session.status)
        self.assertEqual('plan_conditions_met', session.last_verification.evidence_kind)
        # The committed plan is shown to the model on later observations, never a hidden checklist.
        self.assertEqual(plan.prompt_view(), adapter.contexts[-1]['task_plan'])

    def test_without_proposal_the_goal_itself_becomes_the_single_condition(self):
        loop, session, _ = self.start([(scene(0), decision('home')), (scene(1), decision())], goal='回到桌面')
        self.assertEqual(GOAL_FALLBACK_SOURCE, session.task_plan.source)
        self.assertEqual(('c1',), session.task_plan.condition_ids)
        self.assertIn('回到桌面', session.task_plan.success_conditions[0].description)
        loop.run_autonomous_safe_loop(session)
        self.assertEqual('succeeded', session.status)

    def test_later_proposals_do_not_silently_change_the_frozen_plan(self):
        rows = [(scene(0), proposing(decision('home'), CONDITIONS)),
                (scene(1), proposing(finishing(MET), ('只要看到桌面',)))]
        loop, session, _ = self.start(rows, goal='打开蓝牙')
        loop.run_autonomous_safe_loop(session)
        self.assertEqual(1, session.task_plan.version)
        self.assertEqual(CONDITIONS, tuple(item.description for item in session.task_plan.success_conditions))

    def test_user_message_allows_one_revised_plan_version(self):
        revised = ('当前页面是WLAN设置', 'WLAN开关显示为开启')
        rows = [(scene(0), proposing(decision('home'), CONDITIONS)),
                (scene(1), proposing(decision('back'), revised)),
                (scene(2), finishing([{'id': 'c1', 'status': 'met', 'evidence': '标题WLAN'},
                                      {'id': 'c2', 'status': 'met', 'evidence': 'WLAN开'}]))]
        loop, session, _ = self.start(rows, goal='打开蓝牙')
        first_id = session.task_plan.plan_id
        session.conversation.append({'role': 'user', 'content': '改成打开WLAN'})
        loop.run_autonomous_safe_loop(session)
        self.assertEqual('succeeded', session.status)
        self.assertEqual((first_id, 2), (session.task_plan.plan_id, session.task_plan.version))
        self.assertEqual(revised, tuple(item.description for item in session.task_plan.success_conditions))


class VerifierPlanConditionTests(LoopHarness):
    def test_finish_with_unmet_condition_reobserves_and_tells_the_model_why(self):
        unmet = [{'id': 'c1', 'status': 'met', 'evidence': '页面标题为蓝牙'},
                 {'id': 'c2', 'status': 'not_met', 'evidence': '开关仍为关闭'}]
        rows = [(scene(0), proposing(decision('home'), CONDITIONS)), (scene(1), finishing(unmet)),
                (scene(2), decision('tap_semantic')), (scene(3), finishing(MET))]
        loop, session, adapter = self.start(rows, goal='打开蓝牙')
        loop.run_autonomous_safe_loop(session, max_physical_actions=1)
        self.assertNotEqual('succeeded', session.status)
        self.assertEqual('plan_conditions_unmet', session.last_verification.evidence_kind)
        self.assertIn('c2', ''.join(session.last_verification.unmet))
        self.assertIsNone(session.snapshot()['last_verification']['terminal_status'])

    def test_model_finish_without_condition_report_never_ends_the_task(self):
        bare_finish = {**decision(), 'condition_results': None}
        loop, session, adapter = self.start([(scene(0), bare_finish)], goal='检查当前页面')
        self.assertEqual('needs_reobservation', session.status)
        self.assertEqual(0, session.physical_actions)
        self.assertEqual([], adapter.calls)

    def test_zero_action_check_task_can_finish_on_first_frame_with_its_own_conditions(self):
        rows = [(scene(0), proposing(finishing([{'id': 'c1', 'status': 'met', 'evidence': '标题为第0页'}]),
                                     ('当前页面标题可见',)))]
        loop, session, adapter = self.start(rows, goal='看一下当前页面标题')
        self.assertEqual('succeeded', session.status)
        self.assertEqual([], adapter.calls)


class ExecutionPermitTests(LoopHarness):
    def test_permit_is_issued_with_policy_and_consumed_exactly_once(self):
        loop, session, adapter = self.start([(scene(0), decision('home')), (scene(1), decision())])
        permit = session.execution_permit
        self.assertEqual((session.task_plan.plan_id, 1), (permit.plan_id, permit.plan_version))
        self.assertEqual('auto_execute', permit.policy_status)
        self.assertEqual(session.confirmation_authority.action_digest, permit.action_digest)
        self.assertTrue(Path(session.run_dir, 'execution_permit_step_1.json').is_file())
        loop.confirm_one(session, session.confirmation_authority.scope())
        self.assertTrue(permit.consumed)
        self.assertEqual(1, len(adapter.calls))

    def test_plan_change_after_issue_invalidates_permit_without_dispatch(self):
        loop, session, adapter = self.start([(scene(0), decision('home')), (scene(1), decision())])
        session.task_plan = TaskPlan.commit(task_id=session.session_id, device_id=session.device_id,
            goal='另一个目标', revision=1, proposed_conditions=('别的条件',), previous=session.task_plan)
        with self.assertRaisesRegex(Exception, '计划版本已变化'):
            loop.confirm_one(session, session.confirmation_authority.scope())
        self.assertEqual([], adapter.calls)
        self.assertEqual(0, session.physical_actions)

    def test_lease_change_after_issue_invalidates_permit_without_dispatch(self):
        loop, session, adapter = self.start([(scene(0), decision('home')), (scene(1), decision())])
        session.execution_permit.lease_epoch += 1
        with self.assertRaisesRegex(Exception, '租约'):
            loop.confirm_one(session, session.confirmation_authority.scope())
        self.assertEqual([], adapter.calls)

    def test_reobservation_revokes_the_unused_permit(self):
        loop, session, _ = self.start([(scene(0), decision('home')), (scene(1), decision())])
        permit = session.execution_permit
        loop._clear_action(session)
        self.assertTrue(permit.consumed)
        self.assertEqual('reobserve', permit.invalid_reason)
        self.assertIsNone(session.execution_permit)

    def test_authentication_permit_needs_the_user_effect_confirmation(self):
        loop, session, adapter = self.start([(scene(0), decision('tap_semantic', meaning='authentication')),
            (scene(1), decision())], goal='登录账号')
        self.assertEqual('requires_user_confirmation', session.execution_permit.policy_status)
        with self.assertRaises(TaskPlanError):
            ExecutionPermit.consume(session.execution_permit, task_id=session.session_id,
                device_id=session.device_id, lease_epoch=session.lease_epoch, revision=session.step_number,
                plan=session.task_plan, observation_id=session.execution_permit.observation_id,
                fingerprint=session.execution_permit.fingerprint,
                intent_digest=session.execution_permit.intent_digest,
                action_digest=session.execution_permit.action_digest, confirmed_effect_ids=())
        self.assertEqual([], adapter.calls)

    def test_terminal_candidates_never_receive_a_permit(self):
        with self.assertRaises(TaskPlanError):
            plan = TaskPlan.commit(task_id='t', device_id='d', goal='g', revision=1, proposed_conditions=None)
            ExecutionPermit.issue(task_id='t', device_id='d', lease_epoch=0, revision=1, plan=plan,
                observation_id='o', fingerprint='f', intent_digest='i', action_digest='a',
                policy_version='p', policy_status='terminal_candidate', effect_ids=())


if __name__ == '__main__':
    unittest.main()
