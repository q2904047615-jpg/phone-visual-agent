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


class DecisionSchemaBranchTests(unittest.TestCase):
    """Regression (2026-11-05 live runs): DashScope emits only the keys of the
    matching anyOf branch, so branches that listed just their action fields
    dropped postcondition from every response and no action task could ever
    be judged successful."""

    def test_every_branch_carries_the_full_decision_object(self):
        from agent.infrastructure.generic_scene_observer import _single_step_response_format
        response_format = _single_step_response_format(
            {'objective': 'x', 'entities': {}}, input_structure_required=False, request_height=1000,
            available_action_kinds=('tap_semantic', 'scroll', 'home', 'swipe_element', 'wait_for_change'))
        decision = response_format['json_schema']['schema']['properties']['decision']
        for branch in decision['anyOf']:
            label = branch['properties']['status'].get('enum'), branch['properties']['action']
            with self.subTest(branch=str(label)):
                for field in ('postcondition', 'reason', 'confidence', 'previous_action_outcome',
                              'state_action_consistent', 'effect', 'target_description'):
                    self.assertIn(field, branch['properties'])
                    self.assertIn(field, branch['required'])
                self.assertFalse(branch['additionalProperties'])
        finish = next(b for b in decision['anyOf'] if b['properties']['status'].get('enum') == ['finish'])
        self.assertEqual('object', finish['properties']['postcondition']['type'])


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



def contradictory_finish():
    """The 2026-11-09 live reply (run 2f88b96f): the screen showed 112+13=125, the
    model said the goal was not met and still answered ``finish`` with
    ``state_action_consistent: false``. It used to be a protocol fault that left
    the task stuck before step one."""
    value = decision(goal_status='not_confirmed')
    value['state_action_consistent'] = False
    return value


class ContradictoryFinishTests(LoopHarness):
    def test_protocol_keeps_a_contradictory_finish_but_still_rejects_such_an_action(self):
        from agent.domain.canonical_action_protocol import (
            CanonicalActionProtocolError, normalize_model_step_decision)
        self.assertIs(False, normalize_model_step_decision(contradictory_finish())['state_action_consistent'])
        action = decision('tap_semantic')
        action['state_action_consistent'] = False
        with self.assertRaisesRegex(CanonicalActionProtocolError, '矛盾'):
            normalize_model_step_decision(action)

    def test_contradictory_finish_is_never_success_even_if_marked_confirmed(self):
        reason = PlatformVerifier._unconfirmed_goal_reason(
            {'status': 'confirmed', 'fact': 'x'}, 3, state_action_consistent=False)
        self.assertIn('state_action_consistent=false', reason)
        for actions in (0, 3):
            with self.subTest(actions=actions):
                self.assertTrue(PlatformVerifier._unconfirmed_goal_reason(
                    None, actions, state_action_consistent=False))

    def test_first_screen_contradiction_is_fed_back_and_the_task_carries_on(self):
        adapter = ReobservingAdapter([
            (scene(0), contradictory_finish()),
            (scene(1), decision('tap_semantic')),
            (scene(2), decision()),
        ])
        loop, session, _ = self.start([], goal='先清空计算器，再计算 12+34', adapter=adapter,
            max_observations=10)
        self.assertEqual('needs_reobservation', session.status)
        self.assertEqual(1, session.rejected_finish_streak)
        loop.run_autonomous_safe_loop(session)
        self.assertEqual('succeeded', session.status)
        self.assertEqual(1, len(adapter.calls))
        fed_back = [ctx.get('verifier_feedback') for ctx in adapter.contexts if ctx.get('verifier_feedback')]
        self.assertTrue(fed_back)
        self.assertIn('state_action_consistent=false', fed_back[0])

    def test_repeated_contradictory_finish_pauses_instead_of_looping(self):
        loop, session, adapter = self.start([(scene(0), contradictory_finish())],
            goal='计算 12+34', max_observations=10)
        loop.run_autonomous_safe_loop(session)
        self.assertEqual('paused', session.status)
        self.assertEqual([], adapter.calls)
        self.assertIn('未判定成功', session.auto_pause_reason)


if __name__ == '__main__':
    unittest.main()
