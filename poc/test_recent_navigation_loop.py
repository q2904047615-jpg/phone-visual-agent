"""User-approved local navigation; no model network, camera or phone actions."""
import json
from unittest.mock import patch
from agent.application.action_adapter import GenericActionAdapterError
from agent.domain.recent_navigation import LOCAL_NAVIGATION_SOURCE
from test_single_visual_loop import Adapter, LoopHarness, scene, decision
from test_raw_whole_task_loop import RawAdapter


class RecentNavigationTests(LoopHarness):
    def rows(self, app='notes', after_home=None):
        return [(scene(0, app=app), decision('open_recent_apps')),
            (scene(1, app='launcher'), after_home or decision()),
            (scene(2, app='system'), decision())]

    def test_raw_reply_controller_loop_orders_home_then_recent_across_apps(self):
        for app in ('notes', 'messenger', 'browser'):
            with self.subTest(app=app):
                a = RawAdapter(self.rows(app))
                loop, s, a = self.start([], adapter=a, goal='查看最近打开的应用')
                loop.run_autonomous_safe_loop(s)
                self.assertEqual(['home', 'open_recent_apps'], [x.action for x in a.calls])
                self.assertEqual(('succeeded', 2, 3), (s.status, s.physical_actions, a.model_calls))
                self.assertEqual([0, 1, 2], [len(x['history']) for x in a.contexts])
                self.assertTrue(all(before is after for before, after in a.scopes))
                self.assertFalse(s.recent_navigation.pending)
                self.assertEqual(['home', 'open_recent_apps'],
                    [x['canonical_action']['action'] for x in a.contexts[-1]['history']])

    def test_new_launcher_frame_overrules_unexecuted_other_action_or_finish(self):
        for payload in (decision(), decision('home'), decision('tap_semantic', meaning='send_message')):
            with self.subTest(payload=payload):
                loop, s, a = self.start(self.rows(after_home=payload))
                loop.run_autonomous_safe_loop(s)
                self.assertEqual(['home', 'open_recent_apps'], [x.action for x in a.calls])
                self.assertEqual('succeeded', s.status)

    def test_already_launcher_does_not_press_home(self):
        loop, s, a = self.start([(scene(0, app='launcher'), decision('open_recent_apps')),
            (scene(1, app='system'), decision())])
        loop.run_autonomous_safe_loop(s)
        self.assertEqual(['open_recent_apps'], [x.action for x in a.calls])

    def test_home_failure_reobserves_and_never_opens_recent_from_app(self):
        rows = [(scene(0), decision('open_recent_apps')),
            (scene(1), decision('open_recent_apps', outcome='unmatched')),
            (scene(2, app='launcher'), decision()), (scene(3, app='system'), decision())]
        loop, s, a = self.start(rows)
        loop.run_autonomous_safe_loop(s)
        self.assertEqual(['home', 'home', 'open_recent_apps'], [x.action for x in a.calls])
        self.assertNotEqual(s.history[0]['after_observation_id'], s.history[1]['after_observation_id'])

    def test_unknown_launcher_fact_cannot_open_recent_and_uses_existing_budget(self):
        loop, s, a = self.start([(scene(0), decision('open_recent_apps')),
            (scene(1, app='unknown'), decision())], max_physical_actions=1)
        loop.run_autonomous_safe_loop(s)
        self.assertEqual(('budget_paused', 1), (s.status, s.physical_actions))
        self.assertEqual(['home'], [x.action for x in a.calls])
        self.assertTrue(s.recent_navigation.pending)
        self.assertEqual(s.session_id, loop.device_registry.active_session(s.device_id))
        a.rows[1:] = [(scene(2, app='launcher'), decision()), (scene(3, app='system'), decision())]
        loop.run_autonomous_safe_loop(s, max_physical_actions=2)
        self.assertEqual(['home', 'open_recent_apps'], [x.action for x in a.calls])
        self.assertEqual(('succeeded', 2), (s.status, s.physical_actions))

    def test_pause_keeps_navigation_but_discards_old_scope(self):
        loop, s, a = self.start(self.rows())
        loop.confirm_one(s, s.confirmation_authority.scope())
        old = s.confirmation_authority
        loop.pause(s)
        self.assertEqual('paused', s.status)
        self.assertTrue(old.consumed)
        self.assertTrue(s.recent_navigation.pending)
        # User changed page while paused: do not replay the formerly staged recents.
        a.rows[1:] = [(scene(10, app='browser'), decision()),
            (scene(11, app='launcher'), decision()), (scene(12, app='system'), decision())]
        loop.run_autonomous_safe_loop(s)
        self.assertEqual(['home', 'home', 'open_recent_apps'], [x.action for x in a.calls])
        self.assertEqual('succeeded', s.status)

    def test_missing_current_target_has_zero_action_and_navigation_is_not_finished(self):
        loop, s, a = self.start(self.rows())
        with patch.object(a, 'execute', side_effect=GenericActionAdapterError(
            '当前新截图不再包含 Qwen 已选目标：测试', physical_actions=0)):
            with self.assertRaises(GenericActionAdapterError):
                loop.confirm_one(s, s.confirmation_authority.scope())
        self.assertEqual([], a.calls)
        self.assertTrue(s.recent_navigation.pending)
        self.assertEqual('needs_reobservation', s.status)
        loop.run_autonomous_safe_loop(s)
        self.assertEqual(['home', 'open_recent_apps'], [x.action for x in a.calls])

    def test_missing_home_capability_does_not_try_recent_from_app(self):
        a = Adapter(self.rows())
        with patch.object(a, 'supported_action_kinds', return_value=frozenset({'open_recent_apps'})):
            with self.assertRaisesRegex(ValueError, 'Home'):
                self.start([], adapter=a)
        self.assertEqual([], a.calls)

    def test_explicit_recent_scope_includes_only_real_navigation_prerequisites(self):
        loop, s, a = self.start([], adapter=RawAdapter(self.rows()), required_action_kind='open_recent_apps')
        self.assertEqual(frozenset({'home', 'open_recent_apps'}), s.observation_action_kinds)
        loop.run_autonomous_safe_loop(s)
        self.assertEqual(['home', 'open_recent_apps'], [x.action for x in a.calls])

    def test_home_capability_alone_cannot_satisfy_explicit_recent_capability(self):
        a = Adapter([(scene(0), decision('home'))])
        with patch.object(a, 'supported_action_kinds', return_value=frozenset({'home'})):
            with self.assertRaisesRegex(Exception, '显式动作能力'):
                self.start([], adapter=a, required_action_kind='open_recent_apps')
        self.assertEqual([], a.calls)

    def test_goal_keywords_do_not_start_navigation_or_prevent_finish(self):
        loop, s, a = self.start([(scene(0), decision())], goal='打开后台清理所有卡片')
        self.assertEqual('succeeded', s.status)  # Existing semantic limitation, not claimed fixed.
        self.assertFalse(s.recent_navigation.pending)
        self.assertEqual([], a.calls)

    def test_handoff_after_recent_does_not_override_next_business_action(self):
        rows = self.rows()[:-1] + [(scene(2, app='system'), decision('tap_semantic')),
            (scene(3, app='notes'), decision())]
        loop, s, a = self.start(rows)
        loop.run_autonomous_safe_loop(s)
        self.assertEqual(['home', 'open_recent_apps', 'tap_semantic'], [x.action for x in a.calls])
        self.assertEqual('qwen_same_response_decision', s.history[-1]['qwen_decision']['decision_source'])

    def test_original_model_choice_and_actual_local_action_are_distinct_evidence(self):
        loop, s, a = self.start(self.rows())
        raw = json.loads((s.run_dir / 'model_binding_step_1.json').read_text(encoding='utf-8'))
        self.assertEqual('open_recent_apps', raw['model_decision']['action'])
        self.assertEqual('home', s.qwen_decision.proposal.action.action)
        self.assertEqual(LOCAL_NAVIGATION_SOURCE, s.snapshot()['qwen_decision']['decision_source'])
        self.assertEqual(1, loop.qwen_observer.status()['local_navigation_action_count'])
        self.assertEqual(0, loop.qwen_observer.status()['model_action_count'])
        self.assertEqual([], s.history)

    def test_post_action_observation_failure_never_continues_navigation(self):
        loop, s, a = self.start(self.rows())
        with patch.object(a, 'execute', side_effect=GenericActionAdapterError('新图不可用', physical_actions=1)):
            with self.assertRaises(GenericActionAdapterError):
                loop.run_autonomous_safe_loop(s)
        self.assertEqual(('failed', 1), (s.status, s.physical_actions))
        self.assertIsNone(loop.device_registry.active_session(s.device_id))

    def test_cancelled_navigation_is_not_inherited_by_new_session(self):
        loop, s, a = self.start(self.rows())
        loop.cancel(s)
        a.rows = [(scene(0), decision('home')), (scene(1, app='launcher'), decision())]
        other = loop.start(session_id='other', raw_goal='回主屏幕', device_id='device-1', run_dir=s.run_dir/'other')
        loop.run_autonomous_safe_loop(other)
        self.assertFalse(other.recent_navigation.pending)
        self.assertEqual(['home'], [x.action for x in a.calls])
