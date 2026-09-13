"""Offline all-image transport checks; no cloud requests or phone actions."""
import base64
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from agent.domain.generic_goal import GenericIntentDraft
from agent.domain.semantic_action import SemanticAction
from agent.domain.universal_action_controller import ResolvedSemanticAction
from agent.domain.vision_model import VisionAgentError
from agent.infrastructure.generic_action_adapter import GenericSingleActionAdapter
from agent.infrastructure.generic_scene_observer import SingleStepGenericSceneObserver
from agent.infrastructure.task_screenshot_history import save_task_frames
from agent.infrastructure.observation_images import local_frame_fingerprint
from contract_tests.observation.test_point_scene_projection import wire, decision
from test_support.generic_scene_observer import SequenceProvider, stable_frames, input_audit_payload, audited_application_input


class TaskScreenshotHistoryTests(unittest.TestCase):
    def context(self, task='task-1'):
        return {'objective': '完成十个不同对象的操作', 'entities': {'task_id': task, 'history': []}}

    def observe(self, observer, directory, frames, paths, task='task-1', device='device-1'):
        return observer.observe_with_decision(frames=frames, goal_context=self.context(task),
            device_id=device, response_evidence_dir=directory, current_frame_paths=paths)

    def assert_images(self, content, paths):
        images = [base64.b64decode(p['image_url']['url'].split(',')[1])
            for p in content if p['type'] == 'image_url']
        self.assertEqual([Path(p).read_bytes() for p in paths], images)

    def test_all_saved_images_in_order_for_action_and_finish_without_overwriting(self):
        for choice in (decision('home'), decision(None, status='finish')):
            with self.subTest(status=choice['status']), TemporaryDirectory() as tmp:
                directory = Path(tmp)
                frames = stable_frames()
                earlier = save_task_frames(frames, directory, 'before_step_1', 'device-1')
                before = save_task_frames(frames, directory, 'before_step_1', 'device-1')
                current = save_task_frames(frames, directory, 'after_step_1', 'device-1')
                provider = SequenceProvider([wire(choice)])
                observer = SingleStepGenericSceneObserver(provider)
                observed, chosen = self.observe(observer, directory, frames, current)
                self.assertEqual(12, len(set(earlier + before + current)))
                self.assert_images(provider.messages_seen[0][1]['content'], earlier + before + current)
                self.assertEqual(choice['status'], chosen['status'])
                self.assertEqual(local_frame_fingerprint(frames[-1]), observed.fingerprint)
                self.assertEqual(1, provider.calls)
                evidence = json.loads(Path(observer.last_response_evidence_path).read_text(encoding='utf-8'))
                self.assertEqual(['HISTORY'] * 8 + ['CURRENT'] * 4,
                    [i['group'] for i in evidence['task_screenshots']])
                self.assertEqual(12, observer.last_diagnostics['task_screenshot_count'])
                self.assertEqual(4, observer.last_diagnostics['current_screenshot_count'])

    def test_history_dimensions_and_old_input_do_not_change_current_coordinate_or_focus(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            old = [f.resize((540, 960)) for f in stable_frames((200, 30, 30))]
            current = stable_frames((30, 30, 200))
            old_paths = save_task_frames(old, directory, 'old_focused_input', 'device-1')
            paths = save_task_frames(current, directory, 'current', 'device-1')
            payload = wire(decision('home'), audit=input_audit_payload(application_inputs=[
                audited_application_input(text='', focused=None, bounds=[100, 500, 800, 800])]))
            provider = SequenceProvider([payload])
            observer = SingleStepGenericSceneObserver(provider)
            scene, _ = self.observe(observer, directory, current, paths)
            self.assert_images(provider.messages_seen[0][1]['content'], old_paths + paths)
            self.assertEqual(local_frame_fingerprint(current[-1]), scene.fingerprint)
            field = next(e for e in scene.elements if e.role == 'input')
            self.assertIsNone(field.states.get('focused'))
            self.assertEqual('', field.states['value'])
            self.assertEqual(list(current[-1].size), observer.last_diagnostics['request_image_size'])
            prompt = provider.messages_seen[0][1]['content'][0]['text']
            self.assertIn('不能据此填写当前scene、input_structure、焦点、坐标', prompt)
            self.assertIn('不把同一对象的多帧、多次观察或重复操作当成多个对象的完成', prompt)
            self.assertNotIn('不提供旧截图', prompt)

    def test_reusing_observer_separates_tasks_and_rejects_wrong_archive_owner(self):
        with TemporaryDirectory() as tmp:
            frames = stable_frames()
            provider = SequenceProvider([wire(decision('home')), wire(decision('home'))])
            observer = SingleStepGenericSceneObserver(provider)
            for task in ('task-1', 'task-2'):
                directory = Path(tmp) / task
                paths = save_task_frames(frames, directory, 'initial', 'device-1')
                self.observe(observer, directory, frames, paths, task=task)
                self.assert_images(provider.messages_seen[-1][1]['content'], paths)
            for task, device in [('wrong-task', 'device-1'), ('task-2', 'device-2')]:
                with self.assertRaisesRegex(VisionAgentError, '不属于当前'):
                    self.observe(observer, directory, frames, paths, task=task, device=device)
            self.assertEqual(2, provider.calls)

    def test_missing_history_or_missing_current_binding_never_sends_partial_evidence(self):
        for fault in ('missing_file', 'missing_binding', 'reordered_current'):
            with self.subTest(fault=fault), TemporaryDirectory() as tmp:
                directory = Path(tmp)
                frames = stable_frames()
                old = save_task_frames(frames, directory, 'old', 'device-1')
                paths = save_task_frames(frames, directory, 'current', 'device-1')
                provider = SequenceProvider([wire(decision(None, status='finish'))])
                observer = SingleStepGenericSceneObserver(provider)
                if fault == 'missing_file':
                    Path(old[0]).unlink()
                elif fault == 'missing_binding':
                    paths = ()
                else:
                    paths = tuple(reversed(paths))
                with self.assertRaises((OSError, VisionAgentError)):
                    self.observe(observer, directory, frames, paths)
                self.assertEqual(0, provider.calls)

    def test_cloud_failure_keeps_full_evidence_without_retry_or_finish(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            frames = stable_frames()
            old = save_task_frames(frames, directory, 'old', 'device-1')
            paths = save_task_frames(frames, directory, 'current', 'device-1')
            provider = SequenceProvider([VisionAgentError('context length exceeded')])
            observer = SingleStepGenericSceneObserver(provider)
            with self.assertRaisesRegex(VisionAgentError, 'context length exceeded'):
                self.observe(observer, directory, frames, paths)
            self.assert_images(provider.messages_seen[0][1]['content'], old + paths)
            self.assertEqual(1, provider.calls)
            self.assertEqual(8, observer.last_diagnostics['task_screenshot_count'])

    def test_real_adapter_includes_initial_confirmation_and_post_action_captures(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            frames = stable_frames()
            provider = SequenceProvider([wire(decision('home')), wire(decision(None, status='finish'))])
            observer = SingleStepGenericSceneObserver(provider)
            executor = Mock()
            executor.execute.return_value = SimpleNamespace(physical_actions=1, transport_result=None,
                hardware_receipt=None, metadata={})
            controller = Mock()
            controller.resolve_one.return_value = ResolvedSemanticAction(node_id='step_1', kind='home')
            controller.verify_after_action.return_value = ()
            adapter = GenericSingleActionAdapter(capture=Mock(return_value=frames[-1]), observer=observer,
                robot=Mock(), controller=controller, device_executor=executor, device_id='device-1',
                frame_interval=0, post_action_settle=0)
            goal = GenericIntentDraft(understood=True, app_id='app', app_name='app',
                objective='回主屏幕', entities={'task_id': 'task-1', 'history': []})
            scene, before, paths, _ = adapter.capture_scene(goal, evidence_dir=directory, prefix='initial')
            with patch.object(adapter, '_arm_physical_execution', return_value=(None, None)):
                result = adapter.execute(requested_action=SemanticAction(node_id='step_1', action='home', params={}),
                    planned_scene=scene, planned_frames=before, goal=goal, confirmed=True,
                    evidence_dir=directory, action_authority=SimpleNamespace(revision=1, effect_ids=()))
            pictures = tuple(p for p in paths if p.endswith('.jpg')) + result.before_frame_paths + result.after_frame_paths
            self.assertEqual(12, len(pictures))
            self.assert_images(provider.messages_seen[1][1]['content'], pictures)
            self.assertEqual(2, provider.calls)
            self.assertEqual(1, executor.execute.call_count)
            self.assertIn('"step":1', provider.messages_seen[1][1]['content'][0]['text'])


if __name__ == '__main__':
    unittest.main()
