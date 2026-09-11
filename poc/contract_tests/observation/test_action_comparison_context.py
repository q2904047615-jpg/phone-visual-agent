"""Temporal request/receipt contracts only: no claim about real Qwen accuracy."""
import json
import inspect
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
from PIL import Image
from agent.application.universal_agent_orchestrator import _model_history
from agent.domain.generic_goal import GenericIntentDraft
from agent.domain.qwen_task_context import execution_history_entry
from agent.domain.semantic_action import SemanticAction
from agent.domain.universal_action_controller import ResolvedSemanticAction
from agent.infrastructure.generic_action_adapter import GenericSingleActionAdapter
from agent.infrastructure.generic_scene_observer import SingleStepGenericSceneObserver
from agent.infrastructure.observation_images import local_frame_fingerprint
from agent.infrastructure.dashscope_vision_provider import _image_data_url
from test_support.generic_scene_observer import (
    SequenceProvider,
    stable_frames,
    audited_application_input,
    input_audit_payload,
)
from contract_tests.observation.test_point_scene_projection import wire, decision
from test_single_visual_loop import scene


class ActionComparisonContextTests(unittest.TestCase):
    def history(self, meaning='open_chat_session', step=2):
        return execution_history_entry(step=step,
            requested_action={'action': 'tap_semantic', 'params': {'target': meaning,
                'label': '目标会话', 'target_role': 'list_item'}},
            resolved_action={'kind': 'tap_semantic', 'normalized_point': [.5, .266]},
            physical_actions=1, transport_outcome='executed')

    def context(self, history=None, task_id='task-1'):
        return {'objective': '给目标发送aaazjie？你好',
            'entities': {'task_id': task_id, 'history': [] if history is None else history}}

    def observe(self, *, history=None, focused=None, choice=None):
        frames = stable_frames()
        payload = wire(choice or decision('home'), audit=input_audit_payload(application_inputs=[
            audited_application_input(text='', focused=focused, bounds=[100, 500, 800, 800])]))
        payload['coordinate_space'] = {'kind': 'axis_grid', 'width': 1000, 'height': 1000}
        provider = SequenceProvider([payload])
        observer = SingleStepGenericSceneObserver(provider)
        with tempfile.TemporaryDirectory() as directory:
            observed, result = observer.observe_with_decision(frames=frames,
                goal_context=self.context(history), device_id='device-1',
                response_evidence_dir=Path(directory))
            evidence = json.loads(Path(observer.last_response_evidence_path).read_text(encoding='utf-8'))
        return provider, observer, observed, result, frames, evidence

    def test_shared_history_preserves_semantics_and_copies_nested_params(self):
        for meaning in ('open_chat_session', 'send_message', 'publish_content'):
            with self.subTest(meaning=meaning):
                requested = SemanticAction(node_id='actual', action='tap_semantic',
                    params={'target': meaning, 'label': '目标', 'text': 'aaazjie？你好'})
                resolved = ResolvedSemanticAction(node_id='actual', kind='tap_semantic')
                goal = GenericIntentDraft(understood=True, app_id='app', app_name='应用',
                    objective='原任务', entities={'history': []})
                immediate = GenericSingleActionAdapter._post_action_goal(goal,
                    authority=SimpleNamespace(revision=2), requested=requested,
                    resolved=resolved, physical_actions=1).entities['history']
                refreshed = _model_history(SimpleNamespace(history=[{'step_number': 2,
                    'execution': {'requested_action': requested.to_dict(),
                        'resolved_action': resolved.to_dict(), 'physical_actions': 1,
                        'action_outcome': 'executed'}}]))
                self.assertEqual(immediate, refreshed)
                self.assertEqual(meaning, immediate[0]['canonical_action']['params']['target'])
                requested.params['label'] = 'mutated'
                self.assertEqual('目标', immediate[0]['canonical_action']['params']['label'])
                self.assertEqual([], goal.entities['history'])

    def test_only_current_group_and_text_history_one_call(self):
        p, o, current, _, frames, evidence = self.observe(history=[self.history()])
        content = p.messages_seen[0][1]['content']
        images = [x['image_url']['url'] for x in content if x['type'] == 'image_url']
        self.assertEqual(1, p.calls)
        self.assertEqual([_image_data_url(x) for x in frames[-3:]], images)
        self.assertEqual(local_frame_fingerprint(frames[-1]), current.fingerprint)
        self.assertEqual(4, len(o.last_diagnostics['frame_sharpness_scores']))
        self.assertNotIn('comparison_reference', evidence)
        self.assertNotIn('BEFORE_PREVIOUS_ACTION', str(content))
        self.assertIn('open_chat_session', content[0]['text'])
        self.assertIn('CURRENT', content[1]['text'])
        self.assertEqual('', next(e.states['value'] for e in current.elements if e.role == 'input'))
        self.assertIsNone(next(e.states.get('focused') for e in current.elements if e.role == 'input'))

    def test_initial_and_later_steps_have_no_old_image_entrypoint(self):
        parameters = inspect.signature(SingleStepGenericSceneObserver.observe_with_decision).parameters
        self.assertNotIn('previous_action_frame', parameters)
        self.assertNotIn('previous_action_step', parameters)
        for history in ([], [self.history()], [self.history(step=1), self.history(step=2)]):
            with self.subTest(history=history):
                p, _, _, _, _, evidence = self.observe(history=history)
                self.assertEqual(3, sum(x['type'] == 'image_url' for x in p.messages_seen[0][1]['content']))
                self.assertNotIn('comparison_reference', evidence)

    def test_action_and_state_finish_remain_model_choices(self):
        for choice in (decision('home'), decision(None, status='finish', reason='当前输入已空')):
            with self.subTest(status=choice['status']):
                p, _, _, result, _, _ = self.observe(history=[self.history()],
                    choice=choice)
                self.assertEqual(choice['status'], result['status'])
                prompt = p.messages_seen[0][1]['content'][0]['text']
                self.assertIn('无论history是否为空', prompt)
                self.assertIn('可以无需空清空', prompt)
                self.assertIn('canonical_action', prompt)
                self.assertIn('不能当作本次产生了新效果', prompt)

    def test_reused_observer_sends_only_latest_capture(self):
        payloads = [wire(decision('home')), wire(decision('home'))]
        for payload in payloads:
            payload['coordinate_space'] = {'kind': 'axis_grid', 'width': 1000, 'height': 1000}
        provider = SequenceProvider(payloads)
        observer = SingleStepGenericSceneObserver(provider)
        first = stable_frames((200, 30, 30))
        second = stable_frames((30, 30, 200))
        for frames, history in ((first, []), (second, [self.history()])):
            observer.observe_with_decision(frames=frames, goal_context=self.context(history))
        self.assertEqual(2, provider.calls)
        content = provider.messages_seen[1][1]['content']
        images = [x['image_url']['url'] for x in content if x['type'] == 'image_url']
        self.assertEqual([_image_data_url(x) for x in second[-3:]], images)
        self.assertNotIn(_image_data_url(first[-1]), images)
        self.assertIn('open_chat_session', content[0]['text'])

    def test_adapter_never_passes_old_images_across_tasks_or_steps(self):
        observer = Mock()
        observer.supports_response_evidence = False
        observer.supports_runtime_action_contract = False
        observer.observe_with_decision.return_value = (scene(1), {'status': 'finish'})
        adapter = GenericSingleActionAdapter(capture=Mock(), observer=observer,
            robot=Mock(), device_id='device-1')
        self.assertFalse(hasattr(adapter, '_previous_action_reference'))
        for task, history, expected in [('task-1', [self.history()], True),
                ('task-1', [self.history()], True),  # Fresh re-observation keeps last action comparison.
                ('task-2', [self.history()], False), ('task-1', [], False),
                ('task-1', [self.history(step=3)], False)]:
            adapter._observe_scene([], self.context(history, task))
            self.assertNotIn('previous_action_frame', observer.observe_with_decision.call_args.kwargs)
            self.assertEqual(self.context(history, task), observer.observe_with_decision.call_args.kwargs['goal_context'])

    def test_execution_and_failure_do_not_cache_old_images(self):
        observer = Mock()
        executor = Mock()
        executor.execute.return_value = SimpleNamespace(physical_actions=1, transport_result=None,
            hardware_receipt=None, metadata={})
        controller = Mock()
        controller.resolve_one.return_value = ResolvedSemanticAction(node_id='actual', kind='home')
        adapter = GenericSingleActionAdapter(capture=Mock(), observer=observer, robot=Mock(),
            device_executor=executor, controller=controller, device_id='device-1')
        goal = GenericIntentDraft(understood=True, app_id='app', app_name='应用',
            objective='回到主屏幕', entities={'task_id': 'task-1', 'history': []})
        for step, color in [(1, 'red'), (2, 'blue')]:
            frames = stable_frames((200, 30, 30) if color == 'red' else (30, 30, 200))
            authority = SimpleNamespace(task_id='task-1', revision=step, effect_ids=())
            after = (scene(2), frames, (), (), (), (), {'status': 'finish'})
            with patch.object(adapter, '_capture_confirmation_frames', return_value=(frames, ())), \
                    patch.object(adapter, '_arm_physical_execution', return_value=(None, None)), \
                    patch.object(adapter, '_observe_stable_post_action_scene', return_value=after):
                adapter.execute(requested_action=SemanticAction(node_id='actual', action='home', params={}),
                    planned_scene=scene(1), goal=goal, confirmed=True, planned_frames=frames,
                    action_authority=authority)
                self.assertFalse(hasattr(adapter, '_previous_action_reference'))
                executor.execute.side_effect = RuntimeError('device offline')
                with self.assertRaisesRegex(RuntimeError, 'device offline'):
                    adapter.execute(requested_action=SemanticAction(node_id='actual', action='home', params={}),
                        planned_scene=scene(1), goal=goal, confirmed=True, planned_frames=frames,
                        action_authority=SimpleNamespace(task_id='task-1', revision=step+1, effect_ids=()))
                self.assertFalse(hasattr(adapter, '_previous_action_reference'))
                executor.execute.side_effect = None


if __name__ == '__main__':
    unittest.main()
