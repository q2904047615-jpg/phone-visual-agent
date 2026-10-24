"""Runtime plan commit, Verifier plan conditions and execution-permit bindings."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from agent.domain.task_plan import GOAL_FALLBACK_SOURCE, MODEL_PROPOSAL_SOURCE, TaskPlan
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
        self.assertEqual(CONDITIONS, plan.conditions)
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
        self.assertIn('回到桌面', session.task_plan.conditions[0])
        loop.run_autonomous_safe_loop(session)
        self.assertEqual('succeeded', session.status)

    def test_later_proposals_do_not_silently_change_the_frozen_plan(self):
        rows = [(scene(0), proposing(decision('home'), CONDITIONS)),
                (scene(1), proposing(finishing(MET), ('只要看到桌面',)))]
        loop, session, _ = self.start(rows, goal='打开蓝牙')
        loop.run_autonomous_safe_loop(session)
        self.assertEqual(1, session.task_plan.version)
        self.assertEqual(CONDITIONS, session.task_plan.conditions)

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
        self.assertEqual(revised, session.task_plan.conditions)


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
    def test_authority_carries_plan_lease_and_policy_and_is_consumed_once(self):
        loop, session, adapter = self.start([(scene(0), decision('home')), (scene(1), decision())])
        authority = session.confirmation_authority
        self.assertEqual((session.task_plan.plan_id, 1, 'auto_execute'),
            (authority.plan_id, authority.plan_version, authority.policy_status))
        self.assertNotIn('plan_id', authority.scope())  # client scope wire unchanged
        loop.confirm_one(session, authority.scope())
        self.assertTrue(authority.consumed)
        self.assertEqual(1, len(adapter.calls))

    def test_changed_plan_or_lease_dispatches_nothing_and_reobserves(self):
        for name in ('plan', 'lease'):
            with self.subTest(name):
                loop, session, adapter = self.start([(scene(0), decision('home')), (scene(1), decision())])
                scope = session.confirmation_authority.scope()
                if name == 'plan':
                    session.task_plan = TaskPlan.commit(task_id=session.session_id, device_id=session.device_id,
                        goal='另一个目标', revision=1, proposed_conditions=('别的条件',), previous=session.task_plan)
                else:
                    session.lease_epoch += 1
                with self.assertRaisesRegex(Exception, '执行许可无效'):
                    loop.confirm_one(session, scope)
                self.assertEqual([], adapter.calls)
                self.assertEqual(0, session.physical_actions)
                self.assertEqual('needs_reobservation', session.status)
                self.assertIsNone(session.confirmation_authority)
                session.lease_epoch = 0
                loop.device_registry.release('device-1', 'session-1')

    def test_authentication_needs_the_user_effect_confirmation(self):
        loop, session, adapter = self.start([(scene(0), decision('tap_semantic', meaning='authentication')),
            (scene(1), decision())], goal='登录账号')
        authority = session.confirmation_authority
        self.assertEqual('requires_user_confirmation', authority.policy_status)
        problems = authority.permit_problems(plan=session.task_plan, lease_epoch=session.lease_epoch,
            confirmed_effect_ids=())
        self.assertEqual(['登录/付款尚未获得用户确认'], problems)
        self.assertEqual([], authority.permit_problems(plan=session.task_plan, lease_epoch=session.lease_epoch,
            confirmed_effect_ids=('authentication',)))
        self.assertEqual([], adapter.calls)


if __name__ == '__main__':
    unittest.main()
