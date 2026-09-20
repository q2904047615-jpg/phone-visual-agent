"""Fresh observations and one immutable device action scope; offline only."""
from unittest.mock import patch
import unittest
from test_single_visual_loop import LoopHarness, scene, decision
from agent.domain.qwen_task_context import QwenTaskContext
from agent.application.universal_agent_orchestrator import ObservationBridge


class CurrentFrameActionScopeTests(LoopHarness):
    def test_before_and_after_share_one_issued_action_set(self):
        loop,s,a=self.start([(scene(0),decision('home')),(scene(1),decision('back'))])
        with patch.object(a,'execute',wraps=a.execute) as execute:
            issued=s.observation_action_kinds
            loop.confirm_one(s,s.confirmation_authority.scope())
        values=execute.call_args.kwargs
        self.assertIs(issued,values['available_action_kinds'])
        self.assertIs(issued,values['post_action_available_action_kinds'])

    def test_refresh_creates_new_scope_and_does_not_execute_old_point(self):
        loop,s,a=self.start([(scene(0),decision('tap_semantic')),(scene(1),decision('home'))])
        old=s.confirmation_authority
        a.position=1
        loop.refresh_decision(s)
        self.assertTrue(old.consumed)
        self.assertNotEqual(old.fingerprint,s.confirmation_authority.fingerprint)
        self.assertEqual([],a.calls)

    def test_wrong_device_and_action_digest_each_reject_before_action(self):
        for field,value in [('device_id','other-device'),('action_digest','a'*64)]:
            loop,s,a=self.start([(scene(0),decision('home'))])
            scope=s.confirmation_authority.scope()
            scope[field]=value
            with self.subTest(field=field),self.assertRaisesRegex(Exception,'scope'):
                loop.confirm_one(s,scope)
            self.assertEqual([],a.calls)
            loop.device_registry.release(s.device_id,s.session_id)

    def test_new_session_receives_no_previous_task_history(self):
        loop,s,a=self.start([(scene(0),decision('home')),(scene(1),decision())])
        loop.run_autonomous_safe_loop(s)
        a.position=0
        second=loop.start(session_id='session-2',raw_goal='另一个独立任务',device_id='device-1',
            run_dir=s.run_dir / 'second')
        try:
            self.assertEqual([],a.contexts[-1]['history'])
            self.assertEqual('另一个独立任务',ObservationBridge().goal_draft(second).objective)
        finally:
            loop.device_registry.release('device-1','session-2')

    def test_model_context_has_no_hidden_plan_or_effect_checklist(self):
        context=QwenTaskContext(task_id='one',device_id='device-1',revision=1,raw_goal='清空后回桌面')
        self.assertEqual({'task_id','device_id','revision','raw_goal','history','exact_input_text',
            'protocol_version'},set(context.to_dict()))
        for old in ('current_subgoal','goal','effect_intents','transition_receipt'):
            with self.subTest(old=old),self.assertRaises(TypeError):
                QwenTaskContext.from_dict({**context.to_dict(),old:{}})

if __name__ == '__main__':
    unittest.main()
