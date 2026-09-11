"""Transport facts reach Qwen; no actual network, ADB or phone I/O."""
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest
from agent.application.universal_agent_orchestrator import _model_history
from agent.domain.generic_goal import GenericIntentDraft
from agent.domain.qwen_task_context import execution_history_entry
from agent.domain.semantic_action import SemanticAction
from agent.domain.universal_action_controller import ResolvedSemanticAction
from agent.infrastructure.adb_package_launcher import AdbPackageLauncherError
from agent.infrastructure.device_executor import RobotDeviceExecutor
from agent.infrastructure.generic_action_adapter import GenericSingleActionAdapter
from agent.infrastructure.generic_scene_observer import SingleStepGenericSceneObserver, _single_step_observation_prompt
from test_support.generic_scene_observer import (
    SequenceProvider,
    stable_frames,
    input_audit_payload,
)
from contract_tests.observation.test_point_scene_projection import wire, decision
from test_single_visual_loop import scene


FAILURE = {'transport_status': 'error', 'transport_error': 'ADB 包名启动失败：returncode=1'}


class TransportHistoryTests(unittest.TestCase):
    def goal(self):
        return GenericIntentDraft(understood=True, app_id='current_surface', app_name='当前设备',
            objective='打开示例应用', entities={'history': []})

    def pair(self, metadata, kind='launch_app'):
        requested = SemanticAction(node_id='actual', action=kind, params={})
        resolved = ResolvedSemanticAction(node_id='actual', kind=kind)
        immediate = GenericSingleActionAdapter._post_action_goal(self.goal(),
            authority=SimpleNamespace(revision=1), requested=requested, resolved=resolved,
            physical_actions=1, execution_metadata=metadata).entities['history'][0]
        later = _model_history(SimpleNamespace(history=[{'step_number': 1, 'execution': {
            'requested_action': requested.to_dict(), 'resolved_action': resolved.to_dict(),
            'physical_actions': 1, 'action_outcome': 'executed', 'execution_metadata': metadata}}]))[0]
        return immediate, later

    def test_failure_accepted_text_and_unknown_receipts_use_same_projection(self):
        cases = [('launch_app', FAILURE), ('launch_app', {'transport_status': 'accepted'}),
            ('input_verified_text', {'transport_status': 'accepted', 'transport_receipt': {'private': 'not-forwarded'}}),
            ('tap_semantic', {'input_events_dispatched': True}), ('home', {})]
        for kind, metadata in cases:
            with self.subTest(kind=kind, metadata=metadata):
                immediate, later = self.pair(metadata, kind)
                self.assertEqual(immediate, later)
                self.assertEqual(metadata.get('transport_status'), immediate['transport_status'])
                self.assertEqual(metadata.get('transport_error'), immediate['transport_error'])
                self.assertNotIn('transport_receipt', immediate)
                self.assertEqual('executed', immediate['transport_outcome'])
                self.assertIsNone(immediate['visual_outcome'])

    def test_projection_does_not_infer_visual_success_or_forward_other_metadata(self):
        metadata = {**FAILURE, 'shell': 'not-for-model', 'stdout': 'private', 'token': 'private'}
        entry = execution_history_entry(step=1, requested_action={}, resolved_action={},
            physical_actions=1, transport_outcome='executed', visual_outcome='matched',
            execution_metadata=metadata)
        self.assertEqual('matched', entry['visual_outcome'])  # Preserve raw Qwen claim, not local arbitration.
        self.assertEqual(FAILURE['transport_error'], entry['transport_error'])
        metadata['transport_error'] = 'mutated'
        self.assertEqual(FAILURE['transport_error'], entry['transport_error'])
        for key in ('shell', 'stdout', 'token'):
            self.assertNotIn(key, entry)

    def test_missing_or_nontext_diagnostics_remain_unknown(self):
        for metadata in (None, {}, {'transport_status': {}, 'transport_error': ['not a diagnostic string']}):
            with self.subTest(metadata=metadata):
                immediate, later = self.pair(metadata)
                self.assertEqual(immediate, later)
                self.assertIsNone(immediate['transport_status'])
                self.assertIsNone(immediate['transport_error'])

    def test_executor_failure_reaches_actual_post_action_model_call_and_later_history(self):
        for failed in (True, False):
            with self.subTest(failed=failed):
                launcher = Mock()
                if failed:
                    launcher.launch.side_effect = AdbPackageLauncherError(FAILURE['transport_error'], attempted=True)
                executor = RobotDeviceExecutor(Mock(), app_launcher=launcher)
                frames = stable_frames()
                payload = wire(decision('home'), audit=input_audit_payload(application_inputs=[]))
                payload['coordinate_space'] = {'kind': 'normalized_1000', 'width': 1000, 'height': 1000}
                provider = SequenceProvider([payload])
                observer = SingleStepGenericSceneObserver(provider)
                controller = Mock(spec=['resolve_one', 'verify_after_action'])
                controller.verify_after_action.return_value = ()
                controller.resolve_one.return_value = ResolvedSemanticAction(node_id='actual',
                    kind='launch_app', launch_ref='app.sample')
                adapter = GenericSingleActionAdapter(capture=lambda: frames[-1].copy(),
                    observer=observer, robot=Mock(), device_executor=executor, controller=controller, app_launcher=launcher,
                    device_id='device-1', frame_interval=0, post_action_settle=0)
                with patch.object(adapter, '_capture_confirmation_frames', return_value=(frames, ())), \
                        patch.object(adapter, '_arm_physical_execution', return_value=(None, None)), \
                        patch.object(observer, 'observe_with_decision', wraps=observer.observe_with_decision) as observe:
                    result = adapter.execute(requested_action=SemanticAction(node_id='actual', action='launch_app', params={}),
                        planned_scene=scene(1), planned_frames=frames, goal=self.goal(), confirmed=True,
                        action_authority=SimpleNamespace(revision=1, effect_ids=()),
                        available_action_kinds=frozenset({'launch_app', 'home'}))
                immediate = observe.call_args.kwargs['goal_context']['entities']['history'][-1]
                later = _model_history(SimpleNamespace(history=[{'step_number': 1, 'execution': result.to_dict()}]))[-1]
                expected = 'error' if failed else 'accepted'
                self.assertEqual(expected, immediate['transport_status'])
                self.assertEqual(expected, later['transport_status'])
                self.assertEqual(immediate['transport_error'], later['transport_error'])
                self.assertEqual(FAILURE['transport_error'] if failed else None, immediate['transport_error'])
                request_prompt = provider.messages_seen[0][1]['content'][0]['text']
                self.assertIn('"transport_status":"' + expected + '"', request_prompt)
                if failed:
                    self.assertIn(FAILURE['transport_error'], request_prompt)
                self.assertEqual(1, provider.calls)
                self.assertEqual(1, launcher.launch.call_count)
                self.assertEqual('home', result.after_model_decision['action'])
                self.assertEqual('executed', result.action_outcome)

    def test_prompt_distinguishes_launch_transport_from_icon_tap(self):
        prompt = _single_step_observation_prompt({}, include_input_structure=False, image_count=1,
            request_image_size=(720, 1280), available_action_kinds=('launch_app', 'tap_semantic'))
        for rule in ('transport_status', 'transport_error', 'launch_app不是点击图标',
            '点击App图标时选择tap_semantic', 'tap_point必须为null', '不能仅因已尝试就假定成功'):
            self.assertTrue(rule in prompt, msg=rule)


if __name__ == '__main__':
    unittest.main()
