"""Whole-task runtime lifecycle and single-authority error contracts. Offline only."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from agent.application.action_adapter import GenericActionAdapterError
from agent.application.qwen_visual_decision import QwenVisualDecisionObserver
from agent.application.universal_agent_orchestrator import ObservationBridge
from test_single_visual_loop import Adapter, LoopHarness, scene, decision


class CountingQwen(QwenVisualDecisionObserver):
    def __init__(self, fail_on=None, error=None):
        super().__init__(SimpleNamespace(status=lambda: {'model':'offline'}),
            trusted_observation_frame_validator=lambda *_a, **_k: None)
        self.calls = []
        self.fail_on, self.error = fail_on, error
    def decide(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_on == len(self.calls):
            raise self.error or RuntimeError('offline parse failure')
        return super().decide(**kwargs)


class FixtureAdapter(Adapter):
    """Repeating navigation fixture with explicit counters, never a real device."""
    def __init__(self, rows=None, *, device_id='device-1'):
        super().__init__(rows or [(scene(0), decision('tap_semantic')),
            (scene(1), decision('tap_semantic'))])
        self.device_id = device_id
        self.capture_calls = self.execute_calls = 0
        self.execute_error = None
        self.action_authorities = []
    def text_transport_profile(self):
        return replace(super().text_transport_profile(), device_id=self.device_id)
    def capture_scene(self, *args, **kwargs):
        self.capture_calls += 1
        return super().capture_scene(*args, **kwargs)
    def execute(self, **kwargs):
        self.execute_calls += 1
        self.action_authorities.append(kwargs['action_authority'])
        if self.execute_error:
            raise self.execute_error
        if self.position == len(self.rows)-1:
            self.rows.append(self.rows[-1])
        return super().execute(**kwargs)


class RuntimeLifecycleTests(LoopHarness):
    def test_start_only_observes_and_stages_one_action(self):
        q = CountingQwen()
        loop,s,a = self.start([(scene(0),decision('home'))],observer=q)
        self.assertEqual('awaiting_confirmation', s.status)
        self.assertEqual([], a.calls)
        self.assertEqual(1,len(q.calls))
        self.assertEqual(s.raw_goal,q.calls[0]['task_context'].raw_goal)

    def test_finish_on_first_frame_is_model_judgment_without_hidden_checklist(self):
        loop,s,a=self.start([(scene(0),decision())],goal='查看当前状态')
        self.assertEqual('succeeded',s.status)
        self.assertEqual([],a.calls)
        self.assertNotIn('task_graph',s.snapshot())

    def test_whole_task_finish_can_follow_one_action(self):
        loop,s,a=self.start([(scene(0),decision('home')),(scene(1),decision())])
        loop.run_autonomous_safe_loop(s)
        self.assertEqual('succeeded',s.status)
        self.assertEqual(1,len(a.calls))

    def test_navigation_failure_can_choose_new_action_on_new_frame(self):
        loop,s,a=self.start([(scene(0),decision('home')),
            (scene(1),decision('back',outcome='unmatched')),(scene(2),decision())])
        loop.run_autonomous_safe_loop(s)
        self.assertEqual(['home','back'],[x.action for x in a.calls])
        self.assertEqual('succeeded',s.status)

    def test_ordinary_task_never_gets_action_inferred_from_goal_words(self):
        loop,s,a=self.start([(scene(0),decision('tap_semantic'))],goal='先回到主屏幕')
        self.assertIn('tap_semantic',s.observation_action_kinds)
        self.assertIn('home',s.observation_action_kinds)

    def test_explicit_capability_trial_kind_is_the_only_action(self):
        loop,s,a=self.start([(scene(0),decision('home'))],required_action_kind='home')
        self.assertEqual(frozenset({'home'}),s.observation_action_kinds)

    def test_explicit_home_trial_rejects_back_before_execution(self):
        with self.assertRaisesRegex(Exception,'动作集合'):
            self.start([(scene(0),decision('back'))],required_action_kind='home')

    def test_hard_execution_error_counts_attempt_and_releases_lease(self):
        a=FixtureAdapter()
        loop,s,a=self.start([],adapter=a)
        a.execute_error=GenericActionAdapterError('触达不确定',physical_actions=1)
        with self.assertRaises(GenericActionAdapterError):
            loop.run_autonomous_safe_loop(s)
        self.assertEqual(('failed',1,1),(s.status,s.physical_actions,a.execute_calls))
        self.assertIsNone(loop.device_registry.active_session(s.device_id))

    def test_refresh_parse_error_releases_lease_without_action(self):
        q=CountingQwen(fail_on=2)
        loop,s,a=self.start([(scene(0),decision('home'))],observer=q)
        with self.assertRaisesRegex(RuntimeError,'parse failure'):
            loop.refresh_decision(s)
        self.assertEqual('failed',s.status)
        self.assertEqual([],a.calls)
        self.assertIsNone(loop.device_registry.active_session(s.device_id))

    def test_after_action_parse_error_never_becomes_finish(self):
        q=CountingQwen(fail_on=2)
        loop,s,a=self.start([(scene(0),decision('home')),(scene(1),decision())],observer=q)
        with self.assertRaisesRegex(RuntimeError,'parse failure'):
            loop.run_autonomous_safe_loop(s)
        self.assertEqual(('failed',1),(s.status,s.physical_actions))
        self.assertEqual(1,len(a.calls))

    def test_old_scope_cannot_be_replayed_after_next_observation(self):
        loop,s,a=self.start([(scene(0),decision('home')),(scene(1),decision('back'))])
        scope=s.confirmation_authority.scope()
        loop.confirm_one(s,scope)
        with self.assertRaisesRegex(Exception,'scope'):
            loop.confirm_one(s,scope)
        self.assertEqual(1,len(a.calls))

    def test_history_is_factual_and_input_context_is_not_overwritten(self):
        loop,s,a=self.start([(scene(0),decision('home')),(scene(1),decision('back'))],
            goal='  原任务？你好  ')
        loop.confirm_one(s,s.confirmation_authority.scope())
        context=ObservationBridge().goal_draft(s)
        self.assertEqual('  原任务？你好  ',context.objective)
        self.assertEqual(1,len(context.entities['history']))
        self.assertNotIn('active_subgoal_visual_context',context.entities)
        self.assertEqual('home',context.entities['history'][0]['action']['kind'])

    def test_body_already_present_cannot_append_empty_fragment(self):
        with self.assertRaisesRegex(Exception,'prior/fragment/expected'):
            self.start([(scene(0,text='done'),decision('input_verified_text',text='done'))])

    def test_false_model_blocked_is_not_a_terminal_success(self):
        with self.assertRaisesRegex(Exception,'action|finish'):
            self.start([(scene(0),{'status':'blocked','reason':'看不见'})])

    def test_payment_uses_current_step_confirmation(self):
        loop,s,a=self.start([(scene(0),decision('tap_semantic',meaning='financial_transaction')),
            (scene(1),decision())])
        self.assertEqual('awaiting_effect_confirmation',s.status)
        scope=s.effect_confirmation_authority.scope()
        self.assertIn('step_id',scope)
        self.assertNotIn('subgoal_id',scope)
        loop.approve_effects(s,scope)
        self.assertEqual(1,len(a.calls))
        self.assertEqual('succeeded',s.status)

    def test_unmatched_effect_stops_even_if_model_says_whole_task_finish(self):
        loop,s,a=self.start([(scene(0),decision('tap_semantic',meaning='send_message')),
            (scene(1),decision(outcome='unmatched'))])
        loop.run_autonomous_safe_loop(s)
        self.assertEqual('failed',s.status)
        self.assertEqual(1,len(a.calls))


if __name__ == '__main__':
    unittest.main()
