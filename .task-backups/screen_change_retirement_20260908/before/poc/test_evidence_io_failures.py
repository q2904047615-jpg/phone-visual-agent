"""Offline regressions for real-run evidence paths and post-action I/O failures."""
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.infrastructure.atomic_files import atomic_replace_bytes
from agent.infrastructure.generic_scene_observer import SingleStepGenericSceneObserver
from agent.application.action_adapter import GenericActionAdapterError
from test_generic_action_adapter import (
    RawSceneProvider, FakeSceneObserver, FakeRobot, SequenceCapture,
    GenericSingleActionAdapter, SemanticAction, scene, goal,
    FakeAdbKeyboardTextTransport, ConfirmationAuthority, canonical_digest,
    aligned_camera_facts,
)
from test_point_scene_projection import wire
from test_qwen_visual_decision import patterned_frames
import test_universal_agent_orchestrator as loop_fixtures
from test_single_visual_loop import LoopHarness, scene as loop_scene


class EvidenceIOFailureTests(LoopHarness):
    def evidence_directory(self, directory, length=140):
        root = Path(directory).resolve()
        path = root / ('e' * max(1, length - len(str(root)) - 1))
        path.mkdir()
        return path

    def test_atomic_replace_real_run_length_and_short_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.evidence_directory(directory, 120)
            for name in ('report.json', 'qwen_visual_revision_3_' + 'a' * 32
                         + '_after_attempt_1_' + 'b' * 32 + '_model_response.json'):
                with self.subTest(name=name):
                    target = root / name
                    self.assertLess(len(str(target)), 260)
                    atomic_replace_bytes(target, b'old')
                    atomic_replace_bytes(target, b'new')
                    self.assertEqual(b'new', target.read_bytes())
                    self.assertEqual([], list(root.glob('*.tmp')))

    def test_atomic_replace_failure_keeps_original_and_removes_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'report.json'
            atomic_replace_bytes(target, b'original')
            def fail_replace(source, destination):
                self.assertEqual(target.parent, source.parent)
                raise PermissionError('injected replace failure')
            with self.assertRaises(PermissionError):
                atomic_replace_bytes(target, b'new', replace_file=fail_replace)
            self.assertEqual(b'original', target.read_bytes())
            self.assertEqual([target], list(target.parent.iterdir()))

    def test_real_observer_persists_long_after_prefix_and_variation_without_collision(self):
        payload = wire()
        provider = RawSceneProvider([payload] * 3)
        observer = SingleStepGenericSceneObserver(provider)
        with tempfile.TemporaryDirectory() as directory:
            root = self.evidence_directory(directory)
            for prefix in ('qwen_visual_revision_3_' + 'a' * 32 + '_after_attempt_1',
                           'x' * 96, 'x' * 96):
                observer.observe_with_decision(frames=patterned_frames(), device_id='device-1',
                    response_evidence_dir=root, response_evidence_prefix=prefix,
                    available_action_kinds=frozenset({'tap_semantic'}))
                saved = json.loads(Path(observer.last_response_evidence_path).read_text(encoding='utf-8'))
                self.assertEqual(payload, json.loads(saved['redacted_raw_response']))
                self.assertEqual(prefix, saved['response_evidence_prefix'])
            self.assertEqual(3, len(list(root.glob('*_model_response.json'))))
            self.assertEqual([], list(root.glob('*.tmp')))
        self.assertEqual(3, provider.calls)

    def run_post_action_failure(self, failure, *, diagnostic_failure=False):
        observer = FakeSceneObserver([scene('before'), failure])
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(capture=SequenceCapture(['gray'] * 12),
            observer=observer, robot=robot, frame_interval=0, post_action_settle=0,
            post_action_timeout=1)
        with tempfile.TemporaryDirectory() as directory:
            with patch('agent.infrastructure.generic_action_adapter.persist_observer_failure_diagnostic',
                       side_effect=PermissionError('diagnostic write failure') if diagnostic_failure else None,
                       return_value=()):
                with self.assertRaises(GenericActionAdapterError) as caught:
                    adapter.execute(requested_action=SemanticAction(node_id='step', action='tap_semantic',
                        params={'element_id': 'e1', 'target': 'app_icon'}), planned_scene=scene('planned'),
                        goal=goal(), confirmed=True, evidence_dir=Path(directory))
            self.assertEqual(1, caught.exception.physical_actions)
            self.assertIn(str(failure), str(caught.exception))
            self.assertTrue(caught.exception.evidence)
        self.assertEqual([('tap', 300, 400)], robot.actions)
        self.assertEqual(2, observer.calls)
        return caught.exception

    def test_post_action_io_and_unexpected_failures_keep_executed_count(self):
        for failure in (FileNotFoundError('response temporary path'),
                        PermissionError('response access denied'), OSError('disk full'),
                        ValueError('unexpected post-action failure')):
            with self.subTest(failure=type(failure).__name__):
                self.run_post_action_failure(failure)

    def test_diagnostic_io_failure_does_not_mask_original_or_drop_count(self):
        self.run_post_action_failure(FileNotFoundError('response temporary path'), diagnostic_failure=True)

    def test_clear_transport_receipt_survives_post_action_storage_failure(self):
        before = replace(loop_scene(0,text='draft'),
            camera_alignment=aligned_camera_facts())
        element = before.elements[0]
        action = SemanticAction(node_id='clear', action='clear_verified_text', params={
            'element_id': element.element_id, 'target': element.meaning, 'role': 'input',
            'states': element.states, 'text_transport': 'adb_keyboard',
            'input_field_id': 'current_input', 'prior_input_value': 'draft', 'expected_input_value': ''})
        authority = ConfirmationAuthority(session_id='session-1', task_id='task-1', device_id='test-device',
            revision=3, step_id='clear', effect_ids=(), observation_id='observation-1',
            fingerprint=before.fingerprint, decision_node_id=action.node_id,
            action_digest=canonical_digest(action.to_dict()), consumed=True)
        transport = FakeAdbKeyboardTextTransport()
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(capture=SequenceCapture(['gray'] * 12),
            observer=FakeSceneObserver([before, FileNotFoundError('response temporary path')]),
            robot=robot, text_transport=transport, frame_interval=0, post_action_settle=0,
            post_action_timeout=1)
        with self.assertRaises(GenericActionAdapterError) as caught:
            adapter.execute(requested_action=action, planned_scene=before, goal=goal(),
                confirmed=True, action_authority=authority)
        self.assertEqual(1, caught.exception.physical_actions)
        self.assertEqual('accepted', caught.exception.execution_metadata['transport_status'])
        self.assertEqual(['clear_text'], [call[0] for call in transport.calls])
        self.assertEqual([], robot.actions)

    def test_real_adapter_failure_count_reaches_terminal_session_without_retry(self):
        failure = self.run_post_action_failure(FileNotFoundError('response temporary path'))
        adapter = loop_fixtures.FixtureAdapter()
        adapter.execute_error=failure
        loop,session,adapter=self.start([],adapter=adapter)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(GenericActionAdapterError):
                loop.confirm_one(session, session.confirmation_authority.scope())
            saved = json.loads((session.run_dir / 'session.json').read_text(encoding='utf-8'))
            self.assertEqual(1, saved['physical_actions'])
            self.assertEqual('failed', saved['status'])
            with self.assertRaises(RuntimeError):
                loop.confirm_one(session, {})
        self.assertEqual(1, adapter.execute_calls)
        self.assertIsNone(loop.device_registry.active_session(session.device_id))


if __name__ == '__main__':
    unittest.main()
