from __future__ import annotations
from agent.domain import AgentSessionConflictError
from agent.domain import AgentSessionDeviceMismatchError
from agent.infrastructure import InMemoryAgentSessionRepository
from pathlib import Path
from agent.application import StartUniversalAgentSessionCommand
import tempfile
import unittest
from test_support.agent_modular_monolith import (
    FakeDeviceRegistry,
    FakeOrchestrator,
    FakeSession,
    _BaseAgentSessionApplicationTests,
)


class AgentSessionApplicationTests(_BaseAgentSessionApplicationTests):
    def test_start_use_case_owns_task_boundary_and_session_registration(self) -> None:
        registry = FakeDeviceRegistry()
        orchestrator = FakeOrchestrator(registry)
        repository = InMemoryAgentSessionRepository()
        ready_calls: list[str] = []
        lease_calls: list[tuple[str, str]] = []
        task_calls: list[str] = []
        service = self._service(
            orchestrator,
            repository,
            ready_calls=ready_calls,
            lease_calls=lease_calls,
            task_calls=task_calls,
        )

        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run"
            result = service.start(
                StartUniversalAgentSessionCommand(
                    session_id="session-001",
                    raw_goal="查看详情",
                    exact_input_text=None,
                    exact_action_kind=None,
                    exact_target_label="",
                    device_id="phone-01",
                    run_dir=run_dir,
                    auto_advance=False,
                )
            )

            self.assertTrue(run_dir.is_dir())
            self.assertIs(result.session, repository.require("session-001"))
        self.assertEqual(["phone-01"], ready_calls)
        self.assertEqual(["phone-01", "phone-01"], registry.calls)
        self.assertEqual(["phone-01"], task_calls)
        self.assertEqual(
            [("enter", "phone-01"), ("exit", "phone-01")],
            lease_calls,
        )
        self.assertEqual(1, len(orchestrator.start_calls))
        self.assertEqual(0, orchestrator.auto_calls)
        self.assertEqual(0, result.automatic_progress["physical_actions"])

    def test_active_device_conflict_stops_before_output_or_model_work(self) -> None:
        registry = FakeDeviceRegistry("existing-session")
        orchestrator = FakeOrchestrator(registry)
        repository = InMemoryAgentSessionRepository()
        ready_calls: list[str] = []
        lease_calls: list[tuple[str, str]] = []
        task_calls: list[str] = []
        service = self._service(
            orchestrator,
            repository,
            ready_calls=ready_calls,
            lease_calls=lease_calls,
            task_calls=task_calls,
        )

        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "must-not-exist"
            with self.assertRaisesRegex(
                AgentSessionConflictError,
                "已有活动任务",
            ):
                service.start(
                    StartUniversalAgentSessionCommand(
                        session_id="session-002",
                        raw_goal="返回上一页",
                        exact_input_text=None,
                        exact_action_kind=None,
                        exact_target_label="",
                        device_id="phone-01",
                        run_dir=run_dir,
                        auto_advance=True,
                    )
                )
            self.assertFalse(run_dir.exists())
        self.assertEqual([], ready_calls)
        self.assertEqual([], lease_calls)
        self.assertEqual([], task_calls)
        self.assertEqual([], orchestrator.start_calls)
        self.assertEqual(0, orchestrator.auto_calls)

    def test_cross_device_refresh_is_rejected_before_readiness_or_observation(self) -> None:
        orchestrator = FakeOrchestrator(FakeDeviceRegistry())
        repository = InMemoryAgentSessionRepository()
        session = FakeSession(
            session_id="session-003",
            device_id="phone-01",
            run_dir=Path("unused"),
        )
        repository.add(session)
        ready_calls: list[str] = []
        lease_calls: list[tuple[str, str]] = []
        service = self._service(
            orchestrator,
            repository,
            ready_calls=ready_calls,
            lease_calls=lease_calls,
            task_calls=[],
        )

        with self.assertRaises(AgentSessionDeviceMismatchError):
            service.refresh(session, requested_device_id="phone-02")

        self.assertEqual([], ready_calls)
        self.assertEqual([], lease_calls)
        self.assertEqual(0, orchestrator.refresh_calls)
        self.assertEqual(0, session.physical_actions)

    def test_active_snapshots_include_every_current_runtime_state_only(self) -> None:
        repository = InMemoryAgentSessionRepository()
        active_statuses = (
            "created",
            "planning",
            "observing",
            "awaiting_effect_confirmation",
            "awaiting_confirmation",
            "executing_one_action",
            "needs_reobservation",
            "budget_paused",
            "paused",
        )
        terminal_or_retired_statuses = (
            "succeeded",
            "blocked",
            "failed",
            "cancelled",
            "paused_after_action",
        )
        for index, status in enumerate((*active_statuses, *terminal_or_retired_statuses)):
            repository.add(
                FakeSession(
                    session_id=f"session-{index}",
                    device_id=f"phone-{index}",
                    run_dir=Path("unused"),
                    status=status,
                )
            )

        self.assertEqual(
            set(active_statuses),
            {item["status"] for item in repository.active_snapshots()},
        )


if __name__ == "__main__":
    unittest.main()
