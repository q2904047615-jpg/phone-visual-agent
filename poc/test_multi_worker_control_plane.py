from __future__ import annotations

import multiprocessing
import queue
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from agent.infrastructure.sqlite_control_plane_repository import SqliteControlPlaneRepository


def _reserve_worker(database: str, session_id: str, gate, results) -> None:
    repository = SqliteControlPlaneRepository(Path(database), seed_local=False)
    try:
        gate.wait(timeout=10)
        reservation = repository.reserve_runtime_quota(
            organization_id="local-org",
            user_id="local-owner",
            session_id=session_id,
            requested_physical_actions=1,
        )
        results.put((session_id, "reserved", reservation["status"]))
    except Exception as exc:  # pragma: no cover - asserted through the process result
        results.put((session_id, type(exc).__name__, str(exc)))
    finally:
        repository.close()


def _identity_worker(database: str, subject: str, user_id: str, gate, results) -> None:
    repository = SqliteControlPlaneRepository(Path(database), seed_local=False)
    try:
        gate.wait(timeout=10)
        response = repository.update_github_identity(
            organization_id="local-org",
            principal_user_id="local-owner",
            subject=subject,
            user_id=user_id,
            expected_revision=0,
            idempotency_key=f"worker-link-{user_id}",
            login=f"{user_id}-gh",
        )
        results.put((user_id, "linked", response[0]))
    except Exception as exc:  # pragma: no cover - asserted through the process result
        results.put((user_id, type(exc).__name__, str(exc)))
    finally:
        repository.close()


class MultiWorkerControlPlaneTests(unittest.TestCase):
    """Test repository contracts across independent Python processes."""

    def _collect(self, processes, results):
        for process in processes:
            process.join(timeout=15)
            self.assertFalse(process.is_alive())
            self.assertEqual(0, process.exitcode)
        items = []
        while True:
            try:
                items.append(results.get_nowait())
            except queue.Empty:
                return items

    def _workspace_tempdir(self) -> Path:
        root = Path(__file__).parent / "output" / ".multi-worker-tests"
        root.mkdir(parents=True, exist_ok=True)
        directory = root / uuid4().hex
        directory.mkdir()
        self.addCleanup(lambda: shutil.rmtree(directory, ignore_errors=True))
        return directory

    def test_processes_share_quota_reservation_fence(self) -> None:
        database = self._workspace_tempdir() / "runtime.sqlite3"
        repository = SqliteControlPlaneRepository(database)
        repository.upsert_quota(
            organization_id="local-org",
            principal_user_id="local-owner",
            idempotency_key="worker-quota",
            request_hash="worker-quota",
            quota_key="daily",
            period="daily",
            max_runs=1,
            max_physical_actions=1,
            max_model_tokens=10,
        )
        repository.close()

        context = multiprocessing.get_context("spawn")
        gate = context.Barrier(2)
        results = context.Queue()
        processes = [
            context.Process(target=_reserve_worker, args=(str(database), f"run-{index}", gate, results))
            for index in (1, 2)
        ]
        for process in processes:
            process.start()
        observed = self._collect(processes, results)

        self.assertEqual(2, len(observed))
        self.assertEqual(1, sum(item[1] == "reserved" for item in observed))
        self.assertEqual(1, sum(item[1] == "RuntimeError" and "quota_exceeded" in item[2] for item in observed))
        repository = SqliteControlPlaneRepository(database, seed_local=False)
        try:
            quota = next(item for item in repository.list_quotas("local-org") if item["quota_key"] == "daily")
            self.assertEqual(1, quota["reserved"]["runs"])
        finally:
            repository.close()

    def test_processes_compete_for_one_github_subject_without_cross_user_binding(self) -> None:
        database = self._workspace_tempdir() / "runtime.sqlite3"
        repository = SqliteControlPlaneRepository(database)
        repository.create_principal(
            organization_id="local-org",
            principal_user_id="local-owner",
            idempotency_key="worker-principal",
            request_hash="worker-principal",
            user_id="worker-viewer",
            role="viewer",
            team_ids=(),
        )
        repository.close()

        context = multiprocessing.get_context("spawn")
        gate = context.Barrier(2)
        results = context.Queue()
        processes = [
            context.Process(
                target=_identity_worker,
                args=(str(database), "90000001", user_id, gate, results),
            )
            for user_id in ("local-owner", "worker-viewer")
        ]
        for process in processes:
            process.start()
        observed = self._collect(processes, results)

        self.assertEqual(2, len(observed))
        self.assertEqual(1, sum(item[1] == "linked" and item[2] == 200 for item in observed))
        self.assertEqual(1, sum(item[1] in {"ValueError", "KeyError"} for item in observed))
        repository = SqliteControlPlaneRepository(database, seed_local=False)
        try:
            identity = repository.github_identity("90000001")
            self.assertIsNotNone(identity)
            self.assertIn(identity["user_id"], {"local-owner", "worker-viewer"})
            self.assertEqual(1, identity["revision"])
        finally:
            repository.close()


if __name__ == "__main__":
    unittest.main()
