"""Cancel must never deadlock the executor.

Regression for 2026-11-05: a cancel sent while the automatic loop was running
took the (global) session repository lock and then waited for the device lock
that the loop held; the loop then needed the repository lock to save its next
step. Both waited forever, every other session request blocked behind the
repository lock, and the executor had to be SIGKILLed. Synthetic I/O, real
orchestrator, real SQLite repository.
"""
from contextlib import nullcontext
import tempfile
import threading
import unittest
from pathlib import Path

from agent.application.universal_agent_sessions import UniversalAgentSessionApplicationService
from agent.infrastructure.sqlite_session_repository import SqliteAgentSessionRepository
from test_single_visual_loop import LoopHarness, decision, scene


class CancelDuringLoopTests(LoopHarness):
    def wire_service(self, loop, session):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        repo = SqliteAgentSessionRepository(Path(tmp.name) / 'runtime.sqlite3')

        def close_unless_deadlocked():
            # A regression must fail the test, not hang the whole suite in cleanup.
            if repo._lock.acquire(timeout=2):
                repo._lock.release()
                repo.close()
        self.addCleanup(close_unless_deadlocked)
        repo.add(session)
        # Production wiring: every runtime snapshot is saved through the repository lock.
        loop.session_persister = lambda s: repo.save(s, event_type='runtime_snapshot')
        service = UniversalAgentSessionApplicationService(
            orchestrator_provider=lambda: loop, sessions=repo, ensure_device_ready=lambda _d: None,
            exclusive_device_session=lambda _d: nullcontext(), begin_new_task=lambda _d: None)
        return repo, service

    @staticmethod
    def thread(target):
        errors = []

        def run():
            try:
                target()
            except Exception as exc:  # surfaced through the assertions below
                errors.append(exc)
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        return worker, errors

    def test_cancel_during_an_action_returns_at_once_and_ends_after_that_action(self):
        loop, session, adapter = self.start([(scene(0), decision('home')),
            (scene(1), decision('back')), (scene(2), decision())])
        repo, service = self.wire_service(loop, session)
        entered, release = threading.Event(), threading.Event()
        original = adapter.execute

        def delayed(**kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError('test release timeout')
            return original(**kwargs)
        adapter.execute = delayed
        worker, loop_errors = self.thread(lambda: loop.run_autonomous_safe_loop(session))
        try:
            self.assertTrue(entered.wait(2))
            canceller, cancel_errors = self.thread(
                lambda: service.cancel(session.session_id, requested_device_id=session.device_id))
            canceller.join(1)
            self.assertFalse(canceller.is_alive(), 'cancel must not wait for the in-flight action or loop')
            self.assertEqual([], cancel_errors)
            self.assertTrue(session.cancel_requested.is_set())
        finally:
            release.set()
        worker.join(5)
        self.assertFalse(worker.is_alive(), 'the loop deadlocked after the cancel request')
        self.assertEqual([], loop_errors)
        self.assertEqual(('cancelled', 1), (session.status, session.physical_actions))
        self.assertEqual(1, len(adapter.calls), 'no action may run after the cancel request')
        self.assertIsNone(loop.device_registry.active_session(session.device_id))
        # Other session requests are not blocked behind the repository lock.
        self.assertTrue(repo._lock.acquire(timeout=2), 'repository lock is still held')
        repo._lock.release()
        with repo.locked(session.session_id) as stored:
            self.assertEqual('cancelled', stored.status)

    def test_cancel_never_waits_for_the_device_lock_while_holding_the_repository_lock(self):
        loop, session, _ = self.start([(scene(0), decision('home')), (scene(1), decision())])
        repo, service = self.wire_service(loop, session)
        self.assertFalse(session.automatic_loop_enabled)
        holding, go_save = threading.Event(), threading.Event()

        def end_of_loop_window():
            # The loop's last steps: device lock held, then a snapshot save.
            with loop.device_registry.device_lock(session.device_id):
                holding.set()
                go_save.wait(5)
                repo.save(session, event_type='runtime_snapshot')
        holder, holder_errors = self.thread(end_of_loop_window)
        self.assertTrue(holding.wait(2))
        canceller, cancel_errors = self.thread(
            lambda: service.cancel(session.session_id, requested_device_id=session.device_id))
        canceller.join(.3)
        go_save.set()
        holder.join(3)
        canceller.join(3)
        self.assertFalse(holder.is_alive() or canceller.is_alive(), 'cancel and the save deadlocked')
        self.assertEqual(([], []), (holder_errors, cancel_errors))
        self.assertEqual('cancelled', session.status)

    def test_cancel_with_nothing_in_flight_terminates_immediately(self):
        loop, session, adapter = self.start([(scene(0), decision('home')), (scene(1), decision())])
        _, service = self.wire_service(loop, session)
        service.cancel(session.session_id, requested_device_id=session.device_id)
        self.assertEqual('cancelled', session.status)
        self.assertFalse(session.cancel_requested.is_set())
        self.assertEqual([], adapter.calls)
        self.assertIsNone(loop.device_registry.active_session(session.device_id))


if __name__ == '__main__':
    unittest.main()
