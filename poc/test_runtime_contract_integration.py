"""Current lifecycle/repository contracts; synthetic I/O, real runtime objects."""
import threading
from agent.infrastructure.in_memory_session_repository import InMemoryAgentSessionRepository
from agent.infrastructure.capability_acceptance import _confirmation_scope
from test_single_visual_loop import LoopHarness, scene, decision


class RuntimeContractIntegrationTests(LoopHarness):
    def test_real_issued_scope_is_accepted_by_capability_report(self):
        loop, session, _ = self.start([(scene(0), decision('home'))])
        scope = session.confirmation_authority.scope()
        self.assertEqual(scope, _confirmation_scope(scope, session_id=session.session_id,
            task_id=session.session_id, device_id=session.device_id))

    def test_budget_pause_is_visible_in_real_repository(self):
        loop, session, _ = self.start([(scene(0), decision('home'))], max_observations=1)
        repo = InMemoryAgentSessionRepository()
        repo.add(session)
        loop.run_autonomous_safe_loop(session)
        self.assertEqual('budget_paused', session.status)
        self.assertEqual([session.session_id], [x['session_id'] for x in repo.active_snapshots()])

    def test_pause_retains_lease_and_resume_reobserves(self):
        loop, session, adapter = self.start([(scene(0), decision('home')),
            (scene(1), decision())])
        repo = InMemoryAgentSessionRepository()
        repo.add(session)
        old = session.confirmation_authority
        loop.pause(session)
        self.assertEqual('paused', session.status)
        self.assertTrue(old.consumed)
        self.assertIsNone(session.snapshot()['confirmation_scope'])
        self.assertEqual(session.session_id, loop.device_registry.active_session(session.device_id))
        self.assertEqual(1, len(repo.active_snapshots()))
        captures = len(adapter.contexts)
        loop.run_autonomous_safe_loop(session)
        self.assertGreater(len(adapter.contexts), captures + 1)
        self.assertEqual('succeeded', session.status)
        self.assertEqual(1, session.physical_actions)
        self.assertIsNone(loop.device_registry.active_session(session.device_id))

    def test_pause_during_action_is_nonblocking_and_prevents_second_action(self):
        loop, session, adapter = self.start([(scene(0), decision('home')),
            (scene(1), decision('back')), (scene(2), decision())])
        entered, release = threading.Event(), threading.Event()
        original = adapter.execute
        def delayed(**kwargs):
            entered.set()
            if not release.wait(3):
                raise RuntimeError('test release timeout')
            return original(**kwargs)
        adapter.execute = delayed
        errors = []
        def run():
            try:
                loop.run_autonomous_safe_loop(session)
            except Exception as exc:
                errors.append(exc)
        worker = threading.Thread(target=run)
        worker.start()
        try:
            self.assertTrue(entered.wait(2))
            pauser = threading.Thread(target=loop.pause, args=(session,))
            pauser.start()
            pauser.join(.5)
            self.assertFalse(pauser.is_alive(), 'pause must not wait for whole automatic loop')
        finally:
            release.set()
            worker.join(4)
            if 'pauser' in locals():
                pauser.join(4)
        self.assertEqual([], errors)
        self.assertEqual(('paused', 1), (session.status, session.physical_actions))
        self.assertIsNone(session.confirmation_authority)
        loop.cancel(session)
        self.assertEqual([], [x for x in [loop.device_registry.active_session(session.device_id)] if x])

    def test_pause_discards_login_authority_and_requires_fresh_confirmation(self):
        loop, session, adapter = self.start([(scene(0), decision('tap_semantic', meaning='authentication'))])
        authority = session.effect_confirmation_authority
        loop.pause(session)
        self.assertTrue(authority.consumed)
        self.assertIsNone(session.snapshot()['effect_confirmation_scope'])
        loop.refresh_decision(session)
        self.assertEqual('awaiting_effect_confirmation', session.status)
        self.assertIsNot(authority, session.effect_confirmation_authority)
        self.assertEqual([], adapter.calls)

    def test_final_success_and_uncertain_effect_are_not_overwritten_by_pause(self):
        for kind, meaning, outcome, expected in [
            ('home', 'open_details', 'matched', 'succeeded'),
            ('tap_semantic', 'send_message', 'uncertain', 'failed'),
        ]:
            with self.subTest(expected=expected):
                loop, session, adapter = self.start([(scene(0), decision(kind, meaning=meaning)),
                    (scene(1), decision(outcome=outcome))])
                original = adapter.execute
                def request_pause(**kwargs):
                    loop.pause(session)
                    return original(**kwargs)
                adapter.execute = request_pause
                loop.run_autonomous_safe_loop(session)
                self.assertEqual(expected, session.status)
                self.assertEqual(1, session.physical_actions)
                self.assertIsNone(loop.device_registry.active_session(session.device_id))

    def test_real_loop_result_report_writer_and_scope_parser_compose(self):
        from dataclasses import replace
        from PIL import Image
        from test_capability_acceptance import CapabilityAcceptanceCoreTests
        from agent.infrastructure.capability_acceptance import validate_acceptance_report
        from agent.infrastructure.capability_acceptance_runtime import CapabilityAcceptanceManager, CapabilityTrial
        from agent.infrastructure.orientation_safety import OrientationCredential
        fixture = CapabilityAcceptanceCoreTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        loop, session, _ = self.start([(scene(0), decision('home')), (scene(1), decision())])
        before = session.snapshot()
        result = loop.confirm_one(session, session.confirmation_authority.scope())
        self.assertEqual((), result.controller_transition_evidence)
        # Only camera/transport/model I/O is synthetic. Use the actual loop result,
        # authority, report writer and validator, not fake versions of their contracts.
        frame_fixture = fixture._valid_report()
        orientation = frame_fixture['execution']['orientation_credential']
        orientation.update(device_id=session.device_id, scene_fingerprint=result.before_scene.fingerprint)
        before_paths = tuple(frame_fixture['before_frame_paths'])
        after_paths = tuple(frame_fixture['after_frame_paths'])
        result = replace(result, before_frame_paths=before_paths, after_frame_paths=after_paths,
            before_frames=tuple(Image.open(path).convert('RGB') for path in before_paths),
            after_frames=tuple(Image.open(path).convert('RGB') for path in after_paths),
            orientation_credential=OrientationCredential.from_dict(orientation))
        def unused_factory(*_):
            raise AssertionError('This report test must not create a hardware trial or promote a device')
        manager = CapabilityAcceptanceManager(provisional_controller_factory=unused_factory,
            orchestrator_factory=unused_factory, device_registry=loop.device_registry,
            output_dir=fixture.trial_dir, registry_path=fixture.registry_path,
            code_revision_provider=lambda: 'offline-test')
        trial = CapabilityTrial(trial_id='integration-home', candidate_action='home',
            text=session.raw_goal, device_id=session.device_id, run_dir=fixture.trial_dir,
            controller=None, orchestrator=loop, session=session, code_revision='offline-test',
            report_path=fixture.report_path)
        report = manager._write_pass_or_fail_report(trial, before_snapshot=before, result=result)
        self.assertEqual('matched', report['visual_outcome'])
        self.assertEqual(before['confirmation_scope'], report['confirmation_scope'])
        self.assertEqual('passed', validate_acceptance_report(fixture.report_path)['status'])

    def test_scope_consumers_reject_retired_or_malformed_contracts_identically(self):
        loop, session, _ = self.start([(scene(0), decision('home'))])
        scope = session.confirmation_authority.scope()
        variants = [
            {k: v for k, v in scope.items() if k not in {'decision_node_id', 'action_digest'}},
            dict(scope, action_digest='not-a-digest'),
            dict(scope, revision=True),
            dict(scope, surprise='not-part-of-scope'),
        ]
        from agent.application.universal_agent_orchestrator import UniversalAgentOrchestratorError
        from agent.infrastructure.capability_acceptance import CapabilityAcceptanceError
        for changed in variants:
            with self.subTest(scope=changed):
                with self.assertRaises(UniversalAgentOrchestratorError):
                    loop._normalize_confirmation(changed)
                with self.assertRaises(CapabilityAcceptanceError):
                    _confirmation_scope(changed, session_id=session.session_id,
                        task_id=session.session_id, device_id=session.device_id)

    def test_pending_pause_prevents_login_confirmation_execution(self):
        loop, session, adapter = self.start([(scene(0), decision('tap_semantic', meaning='authentication'))])
        authority = session.effect_confirmation_authority
        session.pause_requested.set()
        self.assertIsNone(loop.approve_effects(session, authority.scope()))
        self.assertEqual('paused', session.status)
        self.assertTrue(authority.consumed)
        self.assertEqual([], adapter.calls)

    def test_acceptance_choices_share_actual_physical_capability_mapping(self):
        from agent.domain.action_catalog import PROMOTABLE_ACTION_KINDS
        from agent.domain.action_capabilities import (
            unverified_promotable_actions,
            build_device_capability_snapshot,
        )
        for verified in [[], ['swipe'], ['swipe', 'home', 'back'], ['tap_semantic']]:
            with self.subTest(verified=verified):
                snapshot = build_device_capability_snapshot(device_id='device-1',
                    supported_actions=verified)
                self.assertEqual(sorted(action for action in PROMOTABLE_ACTION_KINDS
                    if not snapshot.actions[action]['enabled']), unverified_promotable_actions(verified))
