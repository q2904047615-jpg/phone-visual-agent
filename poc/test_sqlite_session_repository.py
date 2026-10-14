from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from agent.infrastructure.sqlite_session_repository import SqliteAgentSessionRepository


class _Session:
    def __init__(self, session_id: str = "s1", status: str = "created", physical_actions: int = 0,
                 failed_reason: str = "") -> None:
        self.session_id = session_id
        self.device_id = "device-1"
        self.status = status
        self.physical_actions = physical_actions
        self.failed_reason = failed_reason
        self.run_dir = Path("run")

    def snapshot(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "device_id": self.device_id,
            "status": self.status,
            "physical_actions": self.physical_actions,
            "failed_reason": self.failed_reason,
            "created_at": "2026-10-13T00:00:00+00:00",
        }


class SqliteSessionRepositoryTests(unittest.TestCase):
    def test_snapshot_and_event_are_committed_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteAgentSessionRepository(Path(directory) / "runtime.sqlite3")
            session = _Session()
            repository.add(session)
            session.status = "executing_one_action"
            repository.save(session, event_type="before_dispatch")
            events = repository.events(session.session_id)
            self.assertEqual(["session_created", "before_dispatch"], [item["event_type"] for item in events])
            self.assertEqual("executing_one_action", events[-1]["status"])
            repository.close()

    def test_restart_marks_inflight_session_recovery_required_and_never_replays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "runtime.sqlite3"
            first = SqliteAgentSessionRepository(database)
            first.add(_Session(status="executing_one_action"))
            first.close()

            second = SqliteAgentSessionRepository(database)
            recovered = second.require("s1")
            self.assertEqual("recovery_required", recovered.status)
            self.assertTrue(getattr(recovered, "recovery_required"))
            self.assertEqual("recovery_required", second.events("s1")[-1]["event_type"])
            second.close()

    def test_active_row_written_after_open_is_exposed_as_recovery_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "runtime.sqlite3"
            current_process = SqliteAgentSessionRepository(database)
            external_process = SqliteAgentSessionRepository(database)
            external_process.add(_Session(status="awaiting_confirmation"))
            external_process.close()

            recovered = current_process.require("s1")
            self.assertEqual("recovery_required", recovered.status)
            self.assertTrue(getattr(recovered, "recovery_required"))
            active = current_process.active_snapshots()
            self.assertEqual(["recovery_required"], [item["status"] for item in active])
            self.assertTrue(active[0]["recovery_required"])
            current_process.close()

    def test_cached_live_projection_yields_to_cross_process_terminal_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "runtime.sqlite3"
            current_process = SqliteAgentSessionRepository(database)
            current_process.add(_Session(status="awaiting_confirmation"))
            self.assertEqual("awaiting_confirmation", current_process.require("s1").status)

            external_process = SqliteAgentSessionRepository(database)
            recovered = external_process.reconcile_recovered(
                "s1", requested_device_id="device-1", external_effect="not_dispatched",
            )
            self.assertEqual("cancelled", recovered.status)
            external_process.close()

            self.assertEqual([], current_process.active_snapshots())
            self.assertEqual("cancelled", current_process.require("s1").status)
            current_process.close()

    def test_reconcile_not_dispatched_closes_recovered_session_without_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "runtime.sqlite3"
            first = SqliteAgentSessionRepository(database)
            first.add(_Session(status="awaiting_confirmation"))
            first.close()

            second = SqliteAgentSessionRepository(database)
            recovered = second.reconcile_recovered(
                "s1",
                requested_device_id="device-1",
                external_effect="not_dispatched",
            )
            self.assertEqual("cancelled", recovered.status)
            self.assertFalse(recovered.recovery_required)
            self.assertEqual(0, recovered.physical_actions)
            self.assertEqual("recovery_reconciled", second.events("s1")[-1]["event_type"])
            self.assertEqual([], second.active_snapshots())
            second.close()

    def test_clear_removes_durable_sessions_and_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteAgentSessionRepository(Path(directory) / "runtime.sqlite3")
            repository.add(_Session())
            repository.clear()
            self.assertIsNone(repository.get("s1"))
            self.assertEqual([], repository.active_snapshots())
            repository.close()

    def test_unknown_effect_reconciliation_records_no_replay_compensation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "runtime.sqlite3"
            first = SqliteAgentSessionRepository(database)
            first.add(_Session(status="executing_one_action", physical_actions=1))
            first.close()

            second = SqliteAgentSessionRepository(database)
            recovered = second.reconcile_recovered(
                "s1",
                requested_device_id="device-1",
                external_effect="unknown",
                resolution="unknown_effect_cancelled_no_replay",
            )
            self.assertEqual("cancelled", recovered.status)
            self.assertEqual(1, recovered.physical_actions)
            self.assertEqual("unknown", recovered.recovery_resolution["external_effect"])
            self.assertEqual(
                "unknown_effect_cancelled_no_replay",
                recovered.recovery_resolution["resolution"],
            )
            self.assertEqual("recovery_reconciled", second.events("s1")[-1]["event_type"])
            second.close()

    def test_read_events_is_bounded_and_cursor_replays_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteAgentSessionRepository(Path(directory) / "runtime.sqlite3")
            session = _Session()
            repository.add(session)
            for status in ("observing", "awaiting_confirmation", "cancelled"):
                session.status = status
                repository.save(session, event_type=f"status_{status}")

            first = repository.read_events("s1", limit=2)
            self.assertEqual(2, len(first.events))
            self.assertTrue(first.has_more)
            self.assertEqual(first.events[-1].sequence, first.next_sequence)
            second = repository.read_events("s1", after_sequence=first.next_sequence, limit=2)
            self.assertEqual(2, len(second.events))
            self.assertFalse(second.has_more)
            self.assertEqual(
                [item.sequence for item in first.events + second.events],
                sorted(item.sequence for item in first.events + second.events),
            )
            self.assertEqual((), repository.read_events("s1", after_sequence=second.next_sequence).events)
            repository.close()

    def test_failed_event_exposes_one_report_only_primary_category(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteAgentSessionRepository(Path(directory) / "runtime.sqlite3")
            session = _Session(status="failed", failed_reason="策略确认scope不一致。")
            repository.add(session)
            page = repository.read_events("s1")
            failure = page.events[-1].failure
            self.assertIsNotNone(failure)
            assert failure is not None
            self.assertEqual("policy", failure.category)
            self.assertEqual("runtime", failure.stage)
            self.assertEqual("策略确认scope不一致。", failure.message)
            self.assertEqual("policy", page.to_dict()["events"][-1]["failure"]["category"])
            repository.close()

    def test_replay_cursor_and_limit_reject_ambiguous_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteAgentSessionRepository(Path(directory) / "runtime.sqlite3")
            repository.add(_Session())
            with self.assertRaises(ValueError):
                repository.read_events("s1", after_sequence=-1)
            with self.assertRaises(ValueError):
                repository.read_events("s1", limit=0)
            with self.assertRaises(ValueError):
                repository.read_events("s1", limit=True)  # type: ignore[arg-type]
            repository.close()


if __name__ == "__main__":
    unittest.main()
