import tempfile
import unittest
from agent.domain.execution_budget import ExecutionBudgetExhausted, TaskExecutionBudget
from test_single_visual_loop import LoopHarness, scene, decision
from test_universal_agent_orchestrator import FixtureAdapter
from agent.application.action_adapter import GenericActionAdapterError


class TaskBudgetTests(LoopHarness):
    def test_defaults_and_atomic_configuration_preserve_usage(self):
        budget = TaskExecutionBudget()
        self.assertEqual((100, 200), (budget.max_physical_actions, budget.max_observations))
        budget.request_observation(physical_actions=0)
        with self.assertRaises(ValueError):
            budget.configure(max_physical_actions=300, max_observations=0)
        self.assertEqual(100, budget.max_physical_actions)
        budget.configure(max_physical_actions=300, max_observations=500)
        self.assertEqual(1, budget.observation_attempts)
        for invalid in (True, False, 0, -1, 1.5, '100'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                TaskExecutionBudget(max_observations=invalid)

    def test_no_observation_budget_means_no_action(self):
        budget = TaskExecutionBudget(max_observations=1)
        budget.request_observation(physical_actions=0)
        with self.assertRaises(ExecutionBudgetExhausted):
            budget.request_observation(physical_actions=0, will_execute=True)
        self.assertEqual(1, budget.observation_attempts)

    def start(self, directory=None, *, after_status='action', **budgets):
        rows=[(scene(0),decision('tap_semantic')),
            (scene(1),decision(None if after_status == 'finish' else 'tap_semantic'))]
        return super().start([],adapter=FixtureAdapter(rows),goal='查看详情',**budgets)

    def test_cumulative_across_auto_requests_and_new_scope_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, session, adapter = self.start(directory, max_physical_actions=2, max_observations=10)
            old_authority = session.confirmation_authority
            runner.run_autonomous_safe_loop(session)
            self.assertEqual(('budget_paused', 2, 3),
                (session.status, session.physical_actions, session.execution_budget.observation_attempts))
            self.assertTrue(old_authority.consumed)
            self.assertIsNone(session.confirmation_authority)
            runner.run_autonomous_safe_loop(session)
            self.assertEqual(2, adapter.execute_calls)
            self.assertEqual(4, session.execution_budget.observation_attempts)
            runner.run_autonomous_safe_loop(session, max_physical_actions=3)
            self.assertEqual(3, adapter.execute_calls)
            self.assertEqual(6, session.execution_budget.observation_attempts)
            self.assertEqual('budget_paused', session.status)
            self.assertEqual(3, len({id(a) for a in adapter.action_authorities}))
            self.assertTrue(all(a.consumed for a in adapter.action_authorities))

    def test_final_action_can_use_last_observation_and_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, session, adapter = self.start(directory, after_status='finish',
                max_physical_actions=1, max_observations=2)
            runner.run_autonomous_safe_loop(session)
            self.assertEqual(('succeeded', 1, 2),
                (session.status, adapter.execute_calls, session.execution_budget.observation_attempts))

    def test_initial_observation_is_counted_and_confirm_does_not_bypass(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, session, adapter = self.start(directory, max_observations=1)
            scope = session.confirmation_authority.scope()
            runner.confirm_one(session, scope)
            self.assertEqual(('budget_paused', 0), (session.status, adapter.execute_calls))
            runner.refresh_decision(session)
            self.assertEqual(1, session.execution_budget.observation_attempts)
            self.assertEqual(1, adapter.capture_calls)
            self.assertIsNone(session.confirmation_authority)

    def test_bad_configuration_does_not_fail_or_reset_the_task(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, session, adapter = self.start(directory)
            authority = session.confirmation_authority
            with self.assertRaises(ValueError):
                runner.run_autonomous_safe_loop(session, max_physical_actions=300, max_observations=0)
            self.assertEqual('awaiting_confirmation', session.status)
            self.assertIs(authority, session.confirmation_authority)
            self.assertFalse(authority.consumed)
            self.assertEqual(100, session.execution_budget.max_physical_actions)
            self.assertEqual(1, session.execution_budget.observation_attempts)

    def test_focus_can_be_corrected_more_than_once_using_new_observations(self):
        for focused in (True,False):
            with self.subTest(focused=focused):
                rows=[(scene(i,text='',focused=focused),decision('tap_semantic',role='input',
                    meaning='application_text_input')) for i in range(4)]
                runner,s,adapter=super().start(rows,max_physical_actions=3)
                runner.run_autonomous_safe_loop(s)
                self.assertEqual(('budget_paused',3,4),
                    (s.status,s.physical_actions,s.execution_budget.observation_attempts))
                runner.device_registry.release(s.device_id,s.session_id)

    def test_failed_physical_receipt_is_counted_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, session, adapter = self.start(directory)
            adapter.execute_error = GenericActionAdapterError('执行效果不确定', physical_actions=1)
            with self.assertRaises(GenericActionAdapterError):
                runner.run_autonomous_safe_loop(session)
            self.assertEqual(('failed', 1, 2),
                (session.status, session.physical_actions, session.execution_budget.observation_attempts))
            self.assertEqual(1, adapter.execute_calls)

    def test_read_only_capture_failure_still_counts_an_observation_attempt(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            runner, session, adapter = self.start(directory)
            with patch.object(adapter, 'capture_scene', side_effect=RuntimeError('相机不可用')):
                with self.assertRaises(RuntimeError):
                    runner.refresh_decision(session)
            self.assertEqual(2, session.execution_budget.observation_attempts)
            self.assertEqual(0, session.physical_actions)

    def test_focus_after_matched_input_still_rejects_empty_append(self):
        rows=[(scene(0,text=''),decision('input_verified_text',text='test')),
            (scene(1,text='test'),decision('tap_semantic',role='input',meaning='application_text_input')),
            (scene(2,text='test'),decision('input_verified_text',text='test'))]
        runner,s,a=super().start(rows)
        with self.assertRaisesRegex(Exception,'prior/fragment/expected'):
            runner.run_autonomous_safe_loop(s)
        self.assertEqual('failed',s.status)
        self.assertEqual(['input_verified_text','tap_semantic'],[x.action for x in a.calls])


if __name__ == '__main__':
    unittest.main()
