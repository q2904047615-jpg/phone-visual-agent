from __future__ import annotations

import ast
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from agent.application import (
    StartUniversalAgentSessionCommand,
    UniversalAgentSessionApplicationService,
)
from agent.domain import (
    AgentSessionConflictError,
    AgentSessionDeviceMismatchError,
)
from agent.infrastructure import InMemoryAgentSessionRepository


@dataclass
class FakeSession:
    session_id: str
    device_id: str
    run_dir: Path
    status: str = "awaiting_confirmation"
    physical_actions: int = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "device_id": self.device_id,
            "status": self.status,
            "physical_actions": self.physical_actions,
        }


class FakeDeviceRegistry:
    def __init__(self, active_session_id: str | None = None) -> None:
        self.active_session_id = active_session_id
        self.calls: list[str] = []

    def active_session(self, device_id: str) -> str | None:
        self.calls.append(device_id)
        return self.active_session_id


class FakeOrchestrator:
    def __init__(self, registry: FakeDeviceRegistry) -> None:
        self.device_registry = registry
        self.start_calls: list[dict[str, Any]] = []
        self.refresh_calls = 0
        self.auto_calls = 0
        self.cancel_calls = 0
        self.pause_calls = 0

    def start(self, **kwargs: Any) -> FakeSession:
        self.start_calls.append(dict(kwargs))
        return FakeSession(
            session_id=kwargs["session_id"],
            device_id=kwargs["device_id"],
            run_dir=kwargs["run_dir"],
        )

    def refresh_decision(self, _session: FakeSession) -> object:
        self.refresh_calls += 1
        return object()

    def run_autonomous_safe_loop(
        self,
        _session: FakeSession,
        *,
        max_physical_actions: int,
        max_iterations: int,
    ) -> dict[str, Any]:
        self.auto_calls += 1
        return {
            "physical_actions": 0,
            "iterations": 0,
            "status": "awaiting_confirmation",
            "pause_reason": f"{max_physical_actions}/{max_iterations}",
        }

    def approve_effects(self, _session: FakeSession, _confirmation: Any) -> object:
        return object()

    def confirm_one(self, _session: FakeSession, _confirmation: Any) -> object:
        return object()

    def invalidate_confirmation(self, _session: FakeSession, *, reason: str) -> None:
        return None

    def cancel(self, session: FakeSession) -> None:
        self.cancel_calls += 1
        session.status = "cancelled"

    def pause(self, session: FakeSession) -> None:
        self.pause_calls += 1
        session.status = "paused_after_action"


class AgentSessionApplicationTests(unittest.TestCase):
    def _service(
        self,
        orchestrator: FakeOrchestrator,
        repository: InMemoryAgentSessionRepository,
        *,
        ready_calls: list[str],
        lease_calls: list[tuple[str, str]],
        task_calls: list[str],
    ) -> UniversalAgentSessionApplicationService:
        @contextmanager
        def exclusive(device_id: str) -> Iterator[None]:
            lease_calls.append(("enter", device_id))
            try:
                yield
            finally:
                lease_calls.append(("exit", device_id))

        return UniversalAgentSessionApplicationService(
            orchestrator_provider=lambda: orchestrator,
            sessions=repository,
            ensure_device_ready=ready_calls.append,
            exclusive_device_session=exclusive,
            begin_new_task=task_calls.append,
        )

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


class AgentDependencyBoundaryTests(unittest.TestCase):
    def test_domain_and_application_dependencies_point_inward(self) -> None:
        root = Path(__file__).resolve().parent / "agent"
        banned_by_layer = {
            "domain": {
                "fastapi",
                "pydantic",
                "web_app",
                "agent.application",
                "agent.infrastructure",
                "robot_core",
                "vision_agent",
            },
            "application": {
                "fastapi",
                "pydantic",
                "web_app",
                "agent.infrastructure",
                "robot_core",
                "vision_agent",
            },
            "infrastructure": {
                "fastapi",
                "pydantic",
                "web_app",
                "agent.application",
                "universal_agent_orchestrator",
            },
        }
        violations: list[str] = []
        for layer, banned in banned_by_layer.items():
            for path in (root / layer).glob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    names: list[str] = []
                    if isinstance(node, ast.Import):
                        names = [item.name for item in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        names = [node.module]
                    for name in names:
                        if any(
                            name == item or name.startswith(item + ".")
                            for item in banned
                        ):
                            violations.append(f"{path.name}: {name}")
        self.assertEqual([], violations)

    def test_web_uses_one_session_repository_instead_of_legacy_storage(self) -> None:
        source = (Path(__file__).resolve().parent / "web_app.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("UniversalAgentSessionApplicationService", source)
        self.assertIn("InMemoryAgentSessionRepository", source)
        self.assertNotIn("generic_supervised_sessions", source)
        self.assertNotIn("generic_supervised_session_lock", source)
        self.assertNotIn("_require_generic_session_device", source)

    def test_device_execution_has_one_modular_runtime_entry(self) -> None:
        root = Path(__file__).resolve().parent
        self.assertFalse((root / "device_executor.py").exists())
        self.assertFalse((root / "device_exclusivity.py").exists())

        orchestrator_source = (root / "universal_agent_orchestrator.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("class DeviceTaskRegistry", orchestrator_source)
        self.assertIn(
            "device_registry: DeviceTaskRegistryPort",
            orchestrator_source,
        )
        self.assertNotIn("agent.infrastructure", orchestrator_source)

        forbidden_imports = (
            "from device_executor",
            "import device_executor",
            "from device_exclusivity",
            "import device_exclusivity",
        )
        violations: list[str] = []
        for path in root.rglob("*.py"):
            if path.name.startswith("test_"):
                continue
            source = path.read_text(encoding="utf-8")
            for forbidden in forbidden_imports:
                if forbidden in source:
                    violations.append(f"{path.relative_to(root)}: {forbidden}")
        self.assertEqual([], violations)

    def test_session_evidence_has_one_injected_file_system_implementation(self) -> None:
        root = Path(__file__).resolve().parent
        domain_port = root / "agent" / "domain" / "session_evidence.py"
        infrastructure_store = (
            root
            / "agent"
            / "infrastructure"
            / "file_system_evidence_store.py"
        )
        self.assertTrue(domain_port.is_file())
        self.assertTrue(infrastructure_store.is_file())

        orchestrator_source = (root / "universal_agent_orchestrator.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("class AgentEvidenceStore", orchestrator_source)
        self.assertNotIn("class EvidenceStoreError", orchestrator_source)
        self.assertIn(
            "evidence_store_factory: AgentEvidenceStoreFactory",
            orchestrator_source,
        )
        self.assertNotIn("FileSystemAgentEvidenceStore", orchestrator_source)
        self.assertNotIn("evidence_store_factory or", orchestrator_source)

        web_source = (root / "web_app.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(
            web_source.count(
                "evidence_store_factory=FileSystemAgentEvidenceStore"
            ),
            2,
        )

        forbidden_names = {"AgentEvidenceStore", "EvidenceStoreError"}
        violations: list[str] = []
        for path in root.rglob("*.py"):
            if path.name.startswith("test_"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if node.module != "universal_agent_orchestrator":
                    continue
                for item in node.names:
                    if item.name in forbidden_names:
                        violations.append(
                            f"{path.relative_to(root)}: {item.name}"
                        )
        self.assertEqual([], violations)


if __name__ == "__main__":
    unittest.main(verbosity=2)
