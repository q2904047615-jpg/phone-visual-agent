"""Cross-action regressions after retiring fixed-subgoal authority."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from test_single_visual_loop import LoopHarness, Adapter, scene, decision


class AuditedContractConflictTests(LoopHarness):
    def test_replace_draft_focus_clear_type_send_home_on_variations(self):
        for old,focused in [('旧草稿',True),('old draft',False),('',False)]:
            with self.subTest(old=old,focused=focused):
                rows=[]
                if not focused:
                    rows.append((scene(0,text=old,focused=False),
                        decision('tap_semantic',role='input',meaning='application_text_input')))
                if old:
                    rows.append((scene(1,text=old),decision('clear_verified_text')))
                rows += [(scene(2,text=''),decision('input_verified_text',text='aaazjie？你好')),
                    (scene(3,text='aaazjie？你好'),decision('tap_semantic',meaning='send_message')),
                    (scene(4,text=''),decision('home')),(scene(5,app='launcher'),decision())]
                loop,s,a=self.start(rows)
                loop.run_autonomous_safe_loop(s)
                self.assertEqual('succeeded',s.status)
                self.assertEqual(1,sum(x.action=='input_verified_text' for x in a.calls))
                self.assertEqual(1,sum(x.params.get('target')=='send_message' for x in a.calls))
                self.assertEqual('home',a.calls[-1].action)

    def test_clear_can_be_followed_by_navigation_without_still_focused_requirement(self):
        loop,s,a=self.start([(scene(0,text='draft'),decision('clear_verified_text')),
            (scene(1,text='',focused=None),decision('home')),(scene(2),decision())])
        loop.run_autonomous_safe_loop(s)
        self.assertEqual('succeeded',s.status)
        self.assertEqual(['clear_verified_text','home'],[x.action for x in a.calls])

    def test_next_input_independently_requires_current_focus(self):
        loop,s,a=self.start([(scene(0,text='draft'),decision('clear_verified_text')),
            (scene(1,text='',focused=None),decision('input_verified_text',text='new'))])
        with self.assertRaisesRegex(Exception,'聚焦'):
            loop.run_autonomous_safe_loop(s)
        self.assertEqual(1,len(a.calls))
        self.assertEqual('failed',s.status)

    def test_launch_uses_model_app_name_only_through_local_registry(self):
        class LaunchAdapter(Adapter):
            def supported_action_kinds(self):
                return super().supported_action_kinds() | {'launch_app'}
            def resolve_app_launch_target(self,app_id,app_name):
                self.lookup=(app_id,app_name)
                if app_name=='设置':
                    return SimpleNamespace(launch_ref='trusted-settings',expected_app_id='com.android.settings')
        for app,valid in [('设置',True),('com.attacker.shell',False)]:
            choice={**decision('launch_app'),'app':app}
            adapter=LaunchAdapter([(scene(0),choice)])
            if valid:
                loop,s,a=self.start([],adapter=adapter)
                self.assertEqual(('设置','设置'),a.lookup)
                self.assertEqual('trusted-settings',s.qwen_decision.proposal.action.params['launch_ref'])
                loop.device_registry.release(s.device_id,s.session_id)
            else:
                with self.assertRaisesRegex(Exception,'可信启动映射'):
                    self.start([],adapter=adapter)

    def test_retired_runtime_authorities_cannot_reappear(self):
        root=Path(__file__).resolve().parents[2]
        for name in ('domain/task_graph.py','application/deepseek_task_graph.py',
                'infrastructure/deepseek_intent_provider.py','application/capability_acceptance_planner.py'):
            self.assertFalse((root/'agent'/name).exists())
        source=(root/'agent/application/universal_agent_orchestrator.py').read_text(encoding='utf-8')
        for old in ('deepseek_planner','active_subgoal','forbidden_future_effect_kinds'):
            self.assertNotIn(old,source)

if __name__ == '__main__':
    unittest.main()
