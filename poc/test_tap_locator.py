"""CURRENT-frame tap locator: it may move a tap point, and nothing else.

Regression for the 2026-11-05 real-phone runs: inside the long whole-task
context the model tapped "4" when it meant "2" and "C" when it meant "+".
No model or device I/O.
"""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from PIL import Image

from agent.application.qwen_visual_decision import QwenVisualDecisionObserver
from agent.domain.canonical_action_protocol import CanonicalActionProtocolError, normalize_model_step_decision
from agent.infrastructure import generic_scene_observer as observer_module
from agent.infrastructure.generic_scene_observer import SingleStepGenericSceneObserver
from agent.infrastructure.trusted_observation_frames import (
    build_trusted_observation, validate_trusted_observation_against_frames,
)
from contract_tests.observation.test_point_scene_projection import decision, task_context, wire
from test_canonical_action_protocol import ime_profile
from test_qwen_visual_decision import StatusOnlyProvider, patterned_frames
from test_support.generic_scene_observer import is_tap_locator_request

ALLOWED = frozenset({'tap_semantic', 'dismiss_overlay', 'long_press', 'double_tap', 'scroll', 'home', 'swipe_element'})


class LocatorProvider:
    """Answers the whole-task observation once, then scripted locator replies."""
    configured = True

    def __init__(self, observation, locator_replies):
        self.observation = observation
        self.locator_replies = list(locator_replies)
        self.observation_calls = 0
        self.locator_messages = []

    def status(self):
        return {'configured': True, 'model': 'offline-locator'}

    def chat(self, messages, *, max_tokens, **kwargs):
        if is_tap_locator_request(messages):
            self.locator_messages.append((messages, kwargs))
            reply = self.locator_replies.pop(0)
            if isinstance(reply, BaseException):
                raise reply
            return reply if isinstance(reply, str) else json.dumps(reply)
        self.observation_calls += 1
        return json.dumps(self.observation, ensure_ascii=False)


def frames():
    return [frame.resize((720, 1280), Image.Resampling.NEAREST) for frame in patterned_frames()]


def observe(choice, locator_replies):
    provider = LocatorProvider(deepcopy(wire(choice)), locator_replies)
    observer = SingleStepGenericSceneObserver(provider)
    shots = frames()
    scene, model_decision = observer.observe_with_decision(frames=shots, goal_context={
        'objective': '计算 12+34', 'entities': {'history': []}}, device_id='device-1', available_action_kinds=ALLOWED)
    model_decision.pop('_qwen_reply')
    return provider, observer, scene, model_decision, shots


class TapLocatorTests(unittest.TestCase):
    def test_located_point_replaces_the_drifted_tap_and_reaches_the_bound_action(self):
        provider, observer, scene, choice, shots = observe(
            decision(point=[228, 700], target_description='键盘第四行第二个的数字2键'), [{'point': [410, 790]}])
        self.assertEqual([410.0, 790.0], choice['point'])
        record = observer.last_diagnostics['tap_locator']
        self.assertEqual((True, True, [228, 700], [410.0, 790.0]),
                         (record['called'], record['applied'], record['decision_point'], record['located_point']))
        self.assertEqual('target_description', record['target_source'])
        self.assertEqual(2, observer.last_diagnostics['model_calls'])
        self.assertEqual(1, provider.observation_calls)
        # The locator sees one image (the CURRENT frame) and the target text, nothing else.
        messages, options = provider.locator_messages[0]
        user = messages[1]['content']
        self.assertEqual(1, sum(part['type'] == 'image_url' for part in user))
        self.assertIn('数字2键', ''.join(part.get('text', '') for part in user))
        self.assertEqual({'type': 'json_object'}, options['response_format'])
        trusted = build_trusted_observation(frames=shots, device_id='device-1', scene=scene)
        binder = QwenVisualDecisionObserver(StatusOnlyProvider(),
            trusted_observation_frame_validator=validate_trusted_observation_against_frames)
        result = binder.decide(frames=shots, task_context=task_context(), trusted_observation=trusted,
            model_decision=choice, available_action_kinds=ALLOWED, text_transport_profile=ime_profile(device_id='device-1'))
        self.assertEqual('tap_semantic', result.proposal.action.action)
        # The binder stores the canonical point as a 0..1 fraction of the frame.
        self.assertEqual((0.41, 0.79), tuple(result.proposal.action.params['tap_point']))

    def test_locator_failures_keep_the_decision_point_and_action(self):
        for reply in ({'point': None}, {'point': [1200, 50]}, {'point': ['a', 'b']}, 'not json',
                      TimeoutError('slow')):
            with self.subTest(reply=reply):
                _, observer, _, choice, _ = observe(
                    decision(point=[228, 700], target_description='数字2键'), [reply])
                self.assertEqual([228, 700], choice['point'])
                self.assertEqual('tap', choice['action'])
                record = observer.last_diagnostics['tap_locator']
                self.assertTrue(record['called'])
                self.assertFalse(record['applied'])
                self.assertIn('沿用决策坐标', record['error'])

    def test_missing_description_falls_back_to_the_reason(self):
        provider, observer, _, choice, _ = observe(
            decision(point=[228, 700], reason='下一步点击数字2键'), [{'point': [410, 790]}])
        self.assertEqual([410.0, 790.0], choice['point'])
        self.assertEqual('reason', observer.last_diagnostics['tap_locator']['target_source'])
        user = provider.locator_messages[0][0][1]['content']
        self.assertIn('下一步点击数字2键', ''.join(part.get('text', '') for part in user))

    def test_non_tap_decisions_never_call_the_locator(self):
        for choice in (decision('scroll', direction='down'), decision('home'),
                       decision('swipe_element', start=[500, 800], end=[500, 300]),
                       {**decision(), 'status': 'finish', 'action': None, 'point': None,
                        'postcondition': {'status': 'confirmed', 'fact': '结果已显示'}}):
            with self.subTest(action=choice.get('action')):
                provider, observer, _, _, _ = observe(choice, [])
                self.assertEqual([], provider.locator_messages)
                self.assertFalse(observer.last_diagnostics['tap_locator']['called'])
                self.assertEqual(1, observer.last_diagnostics['model_calls'])

    def test_locator_can_be_switched_off(self):
        with patch.object(observer_module, 'TAP_LOCATOR_ENABLED', False):
            provider, _, _, choice, _ = observe(decision(point=[228, 700], target_description='数字2键'), [])
        self.assertEqual([], provider.locator_messages)
        self.assertEqual([228, 700], choice['point'])


class TargetDescriptionProtocolTests(unittest.TestCase):
    def test_kept_only_for_direct_point_actions(self):
        tap = normalize_model_step_decision(decision(point=[1, 2], target_description='  设置图标  '))
        self.assertEqual('设置图标', tap['target_description'])
        scroll = normalize_model_step_decision(decision('scroll', direction='up', target_description='列表'))
        self.assertIsNone(scroll['target_description'])

    def test_non_string_is_rejected(self):
        with self.assertRaises(CanonicalActionProtocolError):
            normalize_model_step_decision(decision(point=[1, 2], target_description=7))

    def test_schema_requires_a_description_on_tap_branches(self):
        response_format = observer_module._single_step_response_format(
            {'objective': 'x', 'entities': {}}, input_structure_required=False, request_height=1000,
            available_action_kinds=('tap_semantic', 'scroll'))
        decision_schema = response_format['json_schema']['schema']['properties']['decision']
        self.assertIn('target_description', decision_schema['properties'])
        tap_branch = next(b for b in decision_schema['anyOf'] if b['properties']['action'].get('enum') == ['tap'])
        self.assertEqual('string', tap_branch['properties']['target_description']['type'])


if __name__ == '__main__':
    unittest.main()
