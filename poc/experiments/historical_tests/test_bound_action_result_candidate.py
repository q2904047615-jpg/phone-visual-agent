"""Offline candidate contracts. Passing does not establish real model accuracy."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from experiments.bound_action_result_candidate import prepare, project_result_for_evaluation, PREFIX
from agent.domain.action_catalog import CANONICAL_ACTION_KINDS
from agent.infrastructure.generic_scene_observer import _single_step_observation_prompt, _single_step_response_format


class BoundActionResultTests(unittest.TestCase):
    def context(self):
        return {'objective': '发送原文 previous_action_outcome 720×1000 中文？\n  e\u0301 ',
            'entities': {'task_id': 'current', 'history': [
                {'step': 1, 'canonical_action': {'action': 'input_verified_text', 'params': {'text': '原文'}},
                 'action': {'kind': 'input_verified_text'}, 'transport_outcome': 'matched', 'visual_outcome': 'matched'},
                {'step': 2, 'canonical_action': {'action': 'tap_semantic', 'params': {'target': 'send_message'}},
                 'action': {'kind': 'tap_semantic', 'normalized_point': [.88, .898]},
                 'transport_outcome': 'executed', 'visual_outcome': None}]}}

    def prepare(self, context=None, **options):
        return prepare(context or self.context(), request_width=720, request_height=1280,
            image_count=3, available_action_kinds=tuple(sorted(CANONICAL_ACTION_KINDS)), **options)

    def reply(self, action_id, outcome='uncertain', reason='当前画面不能证明触达'):
        return {'decision': {'status': 'action', 'action': 'home', 'reason': '后续返回主屏幕',
            'evaluated_action_id': action_id, 'action_result': outcome, 'action_result_reason': reason}}

    def test_baseline_is_identical_to_production_builder(self):
        ctx = self.context()
        actual = self.prepare(ctx, bind_result=False)
        kinds = tuple(sorted(CANONICAL_ACTION_KINDS))
        self.assertEqual(_single_step_observation_prompt(ctx, include_input_structure=True,
            image_count=3, request_image_size=(720, 1280), available_action_kinds=kinds), actual['prompt'])
        self.assertEqual(_single_step_response_format(ctx, input_structure_required=True,
            request_height=1280, available_action_kinds=kinds), actual['response_format'])

    def test_exact_task_history_and_separate_receipts(self):
        ctx = self.context()
        before = deepcopy(ctx)
        candidate = self.prepare(ctx)
        text = candidate['prompt'].split(PREFIX, 1)[1]
        projected, _ = json.JSONDecoder().raw_decode(text)
        self.assertEqual(ctx['objective'], projected['objective'])
        expected = deepcopy(ctx)
        expected['entities']['history'][0]['transport_outcome'] = 'executed'
        self.assertEqual(expected, projected)
        self.assertEqual(before, ctx)
        self.assertEqual('matched', projected['entities']['history'][0]['visual_outcome'])
        self.assertIsNone(projected['entities']['history'][-1]['visual_outcome'])

    def test_identity_binds_task_step_and_full_last_action_not_earlier_success(self):
        ctx = self.context()
        original = self.prepare(ctx)['expected_action_id']
        ctx['entities']['history'][0]['visual_outcome'] = 'unmatched'
        self.assertEqual(original, self.prepare(ctx)['expected_action_id'])
        for key, value in [('task_id', 'another'), ('history', ctx['entities']['history'][:1])]:
            varied = deepcopy(ctx)
            varied['entities'][key] = value
            self.assertNotEqual(original, self.prepare(varied)['expected_action_id'])
        ctx['entities']['history'][-1]['canonical_action']['params']['target'] = 'publish_content'
        self.assertNotEqual(original, self.prepare(ctx)['expected_action_id'])

    def test_result_fields_are_flat_and_do_not_change_action_schema(self):
        baseline = self.prepare(bind_result=False)['response_format']['json_schema']['schema']
        candidate = self.prepare()['response_format']['json_schema']['schema']
        old = deepcopy(baseline['properties']['decision']['properties'])
        new = deepcopy(candidate['properties']['decision']['properties'])
        old.pop('previous_action_outcome')
        for key in ('evaluated_action_id', 'action_result', 'action_result_reason'):
            new.pop(key)
        self.assertEqual(old, new)
        for key in ('scene', 'input_structure', 'coordinate_space', 'protocol_version'):
            self.assertEqual(baseline['properties'][key], candidate['properties'][key])
        self.assertNotIn('oneOf', json.dumps(candidate))

    def test_initial_observation_has_no_manufactured_effect(self):
        ctx = self.context()
        ctx['entities']['history'] = []
        self.assertIsNone(self.prepare(ctx)['expected_action_id'])
        reply = self.reply(None, None, None)
        self.assertIsNone(project_result_for_evaluation(reply, expected_action_id=None)['decision']['previous_action_outcome'])
        with self.assertRaises(ValueError):
            project_result_for_evaluation(self.reply(None, 'matched'), expected_action_id=None)

    def test_wrong_id_missing_fields_and_legacy_result_are_rejected(self):
        action_id = self.prepare()['expected_action_id']
        replies = [self.reply('old-action'), self.reply(action_id, 'invented'), self.reply(action_id, reason=' ')]
        legacy = self.reply(action_id)
        legacy['decision']['previous_action_outcome'] = 'matched'
        replies.append(legacy)
        missing = self.reply(action_id)
        del missing['decision']['action_result_reason']
        replies.append(missing)
        for reply in replies:
            with self.subTest(reply=reply), self.assertRaises(ValueError):
                project_result_for_evaluation(reply, expected_action_id=action_id)

    def test_semantic_contradiction_is_not_pretended_to_be_locally_solved(self):
        action_id = self.prepare()['expected_action_id']
        reply = self.reply(action_id, 'matched', '更早的输入成功，但最后发送仍未生效')
        # A correct ID does NOT prove this contradictory claim. Real-model
        # evaluation must reject this sample before any production integration.
        self.assertEqual('matched', project_result_for_evaluation(reply,
            expected_action_id=action_id)['decision']['previous_action_outcome'])

    def test_projection_preserves_every_next_action_and_does_not_mutate(self):
        action_id = self.prepare()['expected_action_id']
        for action in (*sorted(CANONICAL_ACTION_KINDS), None):
            reply = self.reply(action_id)
            reply['decision'].update(action=action, status='finish' if action is None else 'action',
                text=' 中文\n  e\u0301 ', tap_point=[880, 1150], start=[123, 456])
            before = deepcopy(reply)
            actual = project_result_for_evaluation(reply, expected_action_id=action_id)['decision']
            for key in ('action', 'status', 'text', 'tap_point', 'start', 'reason'):
                self.assertEqual(reply['decision'][key], actual[key])
            self.assertEqual(before, reply)

    def test_uniform_xy_is_independent_and_preserves_jpeg_dimensions_and_body(self):
        actual = self.prepare(bind_result=False, uniform_xy=True)
        self.assertIn('CURRENT组每张JPEG为720×1280', actual['prompt'])
        self.assertIn('纵坐标都为0..1000', actual['prompt'])
        self.assertEqual([1000], actual['response_format']['json_schema']['schema']['properties']
            ['coordinate_space']['properties']['height']['enum'])
        context, _ = json.JSONDecoder().raw_decode(actual['prompt'].split(PREFIX, 1)[1])
        self.assertEqual(self.context(), context)
        self.assertIn('previous_action_outcome', actual['response_format']['json_schema']['schema']
            ['properties']['decision']['properties'])

    def test_real_failed_send_fixture_binds_send_not_previous_input(self):
        path = Path(__file__).resolve().parents[2] / 'output/web/generic_supervised_20260907_165714_719dd258/0df6ccf437914203937c50b78d9c7721_model_response.json'
        record = json.loads(path.read_text(encoding='utf-8'))
        context = record['goal_context']
        self.assertEqual('send_message', context['entities']['history'][-1]['canonical_action']['params']['target'])
        current = self.prepare(context)
        earlier = deepcopy(context)
        earlier['entities']['history'].pop()
        self.assertNotEqual(current['expected_action_id'], self.prepare(earlier)['expected_action_id'])

    def test_existing_success_empty_clear_and_home_fixtures_prepare_without_future_result(self):
        from experiments.probe_history_dialogue import cases
        _, fixtures = cases()
        self.assertEqual({'old_unsent', 'sent', 'empty_clear', 'sent_then_home'}, set(fixtures))
        for name, fixture in fixtures.items():
            with self.subTest(name=name):
                context = fixture['context']
                before = deepcopy(context)
                for options in ({'bind_result': False}, {},
                                {'bind_result': False, 'uniform_xy': True}):
                    candidate = self.prepare(context, **options)
                    sent, _ = json.JSONDecoder().raw_decode(candidate['prompt'].split(PREFIX, 1)[1])
                    self.assertEqual(context['objective'], sent['objective'])
                    self.assertIsNone(sent['entities']['history'][-1]['visual_outcome'])
                    self.assertEqual('', sent['entities']['history'][-1]['after_scene'])
                self.assertEqual(before, context)

    def test_result_projection_roundtrips_existing_valid_canonical_decisions(self):
        from agent.domain.canonical_action_protocol import normalize_model_step_decision
        from contract_tests.observation.test_point_scene_projection import decision
        action_id = self.prepare()['expected_action_id']
        for kind in ('home', 'back', 'open_recent_apps', 'reveal_system_navigation', None):
            original = decision(kind, status='finish' if kind is None else 'action')
            original['previous_action_outcome'] = 'uncertain'
            original['postcondition'] = {'status': 'unknown', 'fact': '当前截图无法确认上一步动作结果'}
            candidate = deepcopy(original)
            candidate.pop('previous_action_outcome')
            candidate.update(evaluated_action_id=action_id, action_result='uncertain',
                             action_result_reason='当前图不足以确认最后动作效果')
            restored = project_result_for_evaluation({'decision': candidate},
                expected_action_id=action_id)['decision']
            self.assertEqual(normalize_model_step_decision(original), normalize_model_step_decision(restored))


if __name__ == '__main__':
    unittest.main()
