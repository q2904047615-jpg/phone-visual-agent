"""A model finish is a success only when its own CURRENT-frame verdict confirms the goal.

Regression for the 2026-11-05 real-phone run: the task "计算 12+34" ended
``succeeded`` with the calculator showing 514, after the model wrote that the
result was wrong and finished anyway.
"""
import unittest

from agent.application.verifier import PlatformVerifier
from test_single_visual_loop import Adapter, LoopHarness, decision, scene


class ReobservingAdapter(Adapter):
    """Advance to the next scripted row on every re-observation, like a fresh screenshot."""

    def __init__(self, rows):
        super().__init__(rows)
        self.captures = 0

    def capture_scene(self, goal, **kwargs):
        self.captures += 1
        if self.captures > 1:
            self.position += 1
        return super().capture_scene(goal, **kwargs)


class FinishVerdictTests(unittest.TestCase):
    def reason(self, status, actions):
        postcondition = None if status is None else {'status': status, 'fact': '画面事实'}
        return PlatformVerifier._unconfirmed_goal_reason(postcondition, actions)

    def test_model_saying_goal_not_confirmed_is_never_success(self):
        for status in ('not_confirmed', 'unknown'):
            for actions in (0, 3):
                with self.subTest(status=status, actions=actions):
                    self.assertIn(status, self.reason(status, actions))

    def test_after_changing_the_phone_only_confirmed_finishes(self):
        self.assertEqual('', self.reason('confirmed', 6))
        for status in ('not_applicable', None):
            with self.subTest(status=status):
                self.assertIn('confirmed', self.reason(status, 1))

    def test_view_only_answer_without_actions_may_finish_not_applicable(self):
        self.assertEqual('', self.reason('not_applicable', 0))
        self.assertEqual('', self.reason(None, 0))
        self.assertEqual('', self.reason('confirmed', 0))


class FinishVerificationLoopTests(LoopHarness):
    def test_wrong_result_is_rejected_then_corrected_with_feedback(self):
        adapter = ReobservingAdapter([
            (scene(0), decision('tap_semantic')),
            (scene(1), decision(goal_status='not_confirmed')),
            (scene(2), decision('tap_semantic')),
            (scene(3), decision()),
        ])
        loop, session, _ = self.start([], goal='计算 12+34', adapter=adapter, max_observations=10)
        loop.run_autonomous_safe_loop(session)
        self.assertEqual('succeeded', session.status)
        self.assertEqual(2, len(adapter.calls))
        fed_back = [ctx.get('verifier_feedback') for ctx in adapter.contexts if ctx.get('verifier_feedback')]
        self.assertTrue(fed_back, 'the re-observation after a rejected finish must carry the reason')
        self.assertIn('not_confirmed', fed_back[0])
        self.assertEqual('', session.verifier_feedback)
        self.assertEqual(0, session.rejected_finish_streak)

    def test_repeated_unconfirmed_finish_pauses_instead_of_succeeding(self):
        loop, session, adapter = self.start([
            (scene(0), decision('tap_semantic')),
            (scene(1), decision(goal_status='not_confirmed')),
        ], goal='计算 12+34', max_observations=10)
        loop.run_autonomous_safe_loop(session)
        self.assertEqual('paused', session.status)
        self.assertEqual(1, len(adapter.calls))
        self.assertEqual(2, session.rejected_finish_streak)
        self.assertIn('未判定成功', session.auto_pause_reason)
        self.assertEqual('terminal_candidate_rejected', session.last_verification.evidence_kind)
        self.assertIsNone(session.last_verification.terminal_status)

    def test_view_only_question_still_finishes_without_actions(self):
        # The first observation already answers; the session ends during start.
        _loop, session, adapter = self.start([(scene(0), decision(goal_status='not_applicable'))],
            goal='现在屏幕上是什么')
        self.assertEqual('succeeded', session.status)
        self.assertEqual([], adapter.calls)


if __name__ == '__main__':
    unittest.main()
