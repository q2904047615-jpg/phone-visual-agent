from __future__ import annotations

import ast
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from agent.application import (
    CORRECTIVE_RETRY_PROTOCOL_VERSION,
    POST_ACTION_TRANSITION_PROTOCOL_VERSION,
    StartUniversalAgentSessionCommand,
    UniversalAgentSessionApplicationService,
    UniversalAgentSessionState,
)
from agent.domain import (
    CANONICAL_SELECTION_RECEIPT_VERSION,
    AgentSessionConflictError,
    AgentSessionDeviceMismatchError,
    CanonicalSelectionReceipt,
    ConfirmationAuthority,
    EffectConfirmationAuthority,
    VerifiedAppSurfaceLineage,
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
    def test_runtime_session_aggregate_is_owned_by_application(self) -> None:
        class SnapshotAdapter:
            @staticmethod
            def supported_action_kinds() -> frozenset[str]:
                return frozenset({"back", "home"})

            @staticmethod
            def capability_snapshot() -> Any:
                return type(
                    "Capability",
                    (),
                    {"to_dict": lambda self: {"device_id": "phone-1"}},
                )()

        authority = ConfirmationAuthority(
            session_id="session-1",
            task_id="task-1",
            device_id="phone-1",
            revision=1,
            subgoal_id="authenticate",
            effect_ids=("risk-1",),
            observation_id="obs-1",
            fingerprint="frame-1",
            decision_node_id="node-1",
            action_digest="a" * 64,
        )
        session = UniversalAgentSessionState(
            session_id="session-1",
            raw_goal="完成身份认证",
            device_id="phone-1",
            run_dir=Path("unused"),
            adapter=SnapshotAdapter(),
            evidence_store=object(),  # type: ignore[arg-type]
            status="awaiting_confirmation",
            controller_decision=CanonicalSelectionReceipt(
                allowed=True,
                reason="唯一动作已绑定。",
                canonical_class="tap_semantic",
            ),
            confirmation_authority=authority,
            evidence_paths=["a.json", "a.json", "b.json"],
        )

        snapshot = session.snapshot()
        self.assertEqual(1, snapshot["step_number"])
        self.assertEqual(0, snapshot["physical_actions"])
        self.assertTrue(snapshot["confirmation_ready"])
        self.assertEqual(authority.scope(), snapshot["confirmation_scope"])
        self.assertEqual(["a.json", "b.json"], snapshot["evidence"])
        self.assertEqual(["back", "home"], snapshot["available_action_kinds"])
        self.assertEqual(
            CORRECTIVE_RETRY_PROTOCOL_VERSION,
            snapshot["corrective_retry_protocol"],
        )
        self.assertEqual(
            POST_ACTION_TRANSITION_PROTOCOL_VERSION,
            snapshot["post_action_transition_protocol"],
        )
        authority.consumed = True
        consumed = session.snapshot()
        self.assertFalse(consumed["confirmation_ready"])
        self.assertIsNone(consumed["confirmation_scope"])

        root = Path(__file__).resolve().parent
        orchestrator_source = (root / "universal_agent_orchestrator.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("class UniversalAgentSessionState", orchestrator_source)
        self.assertNotIn(
            '"2026-08-16-universal-post-action-transition-v1"',
            orchestrator_source,
        )
        self.assertNotIn(
            '"2026-08-24-fresh-observation-corrective-retry-v1"',
            orchestrator_source,
        )

    def test_verified_app_surface_lineage_is_one_immutable_domain_record(self) -> None:
        lineage = VerifiedAppSurfaceLineage(
            session_id="session-1",
            task_id="task-1",
            device_id="phone-1",
            app_id="messaging-product",
            app_name="Messaging",
            surface_id="conversation",
            source_receipt_id="receipt-1",
            source_subgoal_id="open-conversation",
            functional_foreground_app_id="com.example.messaging",
            physical_actions=2,
        )

        self.assertEqual(
            {
                "session_id": "session-1",
                "task_id": "task-1",
                "device_id": "phone-1",
                "app_id": "messaging-product",
                "app_name": "Messaging",
                "surface_id": "conversation",
                "source_receipt_id": "receipt-1",
                "source_subgoal_id": "open-conversation",
                "functional_foreground_app_id": "com.example.messaging",
                "physical_actions": 2,
            },
            lineage.to_dict(),
        )
        with self.assertRaises(AttributeError):
            lineage.physical_actions = 3  # type: ignore[misc]

        root = Path(__file__).resolve().parent
        orchestrator_source = (root / "universal_agent_orchestrator.py").read_text(
            encoding="utf-8"
        )
        session_source = (
            root / "agent" / "application" / "runtime_session.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("class VerifiedAppSurfaceLineage", orchestrator_source)
        self.assertIn(
            "verified_app_surface_lineage: VerifiedAppSurfaceLineage | None",
            session_source,
        )

    def test_confirmation_authorities_are_domain_scoped_and_stably_sorted(self) -> None:
        action = ConfirmationAuthority(
            session_id="session-1",
            task_id="task-1",
            device_id="phone-1",
            revision=3,
            subgoal_id="authenticate",
            effect_ids=("risk-b", "risk-a"),
            observation_id="obs-1",
            fingerprint="frame-1",
            decision_node_id="node-1",
            action_digest="a" * 64,
        )
        effect = EffectConfirmationAuthority(
            session_id="session-1",
            task_id="task-1",
            device_id="phone-1",
            revision=3,
            subgoal_id="authenticate",
            effect_ids=("risk-b", "risk-a"),
            intent_digest="b" * 64,
            intent_preview={"effect": "authentication"},
        )

        self.assertEqual(["risk-a", "risk-b"], action.scope()["effect_ids"])
        self.assertEqual(["risk-a", "risk-b"], effect.scope()["effect_ids"])
        self.assertNotIn("intent_preview", effect.scope())
        self.assertFalse(action.consumed)
        self.assertFalse(effect.consumed)

        root = Path(__file__).resolve().parent
        orchestrator_source = (root / "universal_agent_orchestrator.py").read_text(
            encoding="utf-8"
        )
        session_source = (
            root / "agent" / "application" / "runtime_session.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("class ConfirmationAuthority", orchestrator_source)
        self.assertNotIn("class EffectConfirmationAuthority", orchestrator_source)
        self.assertIn(
            "confirmation_authority: ConfirmationAuthority | None",
            session_source,
        )
        self.assertIn(
            "effect_confirmation_authority: EffectConfirmationAuthority | None",
            session_source,
        )

    def test_canonical_selection_receipt_is_one_domain_value_object(self) -> None:
        allowed = CanonicalSelectionReceipt(
            allowed=True,
            reason="唯一 canonical 候选已绑定。",
            canonical_class="tap_semantic",
        )
        denied = CanonicalSelectionReceipt(
            allowed=False,
            reason="当前设备不支持该动作。",
        )

        self.assertEqual(
            {
                "allowed": True,
                "reason": "唯一 canonical 候选已绑定。",
                "canonical_class": "tap_semantic",
                "policy_version": CANONICAL_SELECTION_RECEIPT_VERSION,
            },
            allowed.to_dict(),
        )
        self.assertEqual("", denied.to_dict()["canonical_class"])

        root = Path(__file__).resolve().parent
        orchestrator_source = (root / "universal_agent_orchestrator.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("class CanonicalSelectionReceipt", orchestrator_source)
        self.assertNotIn(
            '"2026-08-26-canonical-selection-receipt-v1"',
            orchestrator_source,
        )
        self.assertIn("decision.to_dict()", orchestrator_source)

    def test_domain_and_application_dependencies_point_inward(self) -> None:
        root = Path(__file__).resolve().parent / "agent"
        banned_by_layer = {
            "domain": {
                "canonical_action_protocol",
                "fastapi",
                "pydantic",
                "web_app",
                "agent.application",
                "agent.infrastructure",
                "robot_core",
                "vision_agent",
            },
            "application": {
                "canonical_action_protocol",
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

    def test_camera_coordination_has_one_infrastructure_implementation(self) -> None:
        root = Path(__file__).resolve().parent
        web_source = (root / "web_app.py").read_text(encoding="utf-8")
        infrastructure_source = (
            root / "agent" / "infrastructure" / "camera_coordinator.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("class DeviceCameraCoordinator", web_source)
        self.assertNotIn("class CameraPreviewUnavailable", web_source)
        self.assertNotIn("from io import BytesIO", web_source)
        self.assertIn("class DeviceCameraCoordinator", infrastructure_source)
        self.assertIn("class CameraPreviewUnavailable", infrastructure_source)

    def test_device_controller_registry_has_one_infrastructure_implementation(self) -> None:
        root = Path(__file__).resolve().parent
        web_source = (root / "web_app.py").read_text(encoding="utf-8")
        infrastructure_path = (
            root
            / "agent"
            / "infrastructure"
            / "device_controller_registry.py"
        )
        infrastructure_source = infrastructure_path.read_text(encoding="utf-8")

        self.assertNotIn("class DeviceControllerRegistry", web_source)
        self.assertIn("class DeviceControllerRegistry", infrastructure_source)
        self.assertIn(
            "promotable_actions=PROMOTABLE_ACTIONS",
            web_source,
        )
        self.assertNotIn("capability_acceptance", infrastructure_source)
        self.assertNotIn("universal_agent_orchestrator", infrastructure_source)

        forbidden = (
            "web_app.DeviceControllerRegistry",
            "from web_app import DeviceControllerRegistry",
        )
        violations: list[str] = []
        for path in root.rglob("test_*.py"):
            if path.resolve() == Path(__file__).resolve():
                continue
            source = path.read_text(encoding="utf-8")
            for value in forbidden:
                if value in source:
                    violations.append(f"{path.name}: {value}")
        self.assertEqual([], violations)

    def test_per_device_runtime_resources_are_not_owned_by_web(self) -> None:
        root = Path(__file__).resolve().parent
        web_source = (root / "web_app.py").read_text(encoding="utf-8")
        infrastructure_source = (
            root
            / "agent"
            / "infrastructure"
            / "device_runtime_resources.py"
        ).read_text(encoding="utf-8")

        self.assertIn("class DeviceRuntimeResourceRegistry", infrastructure_source)
        self.assertIn("DeviceRuntimeResourceRegistry(", web_source)
        for legacy in (
            "device_coordination_lock_guard",
            "device_coordination_locks",
            "device_camera_coordinator_guard",
            "device_camera_coordinators",
            "def coordination_lock_for_device",
            "def camera_coordinator_for_device",
        ):
            self.assertNotIn(legacy, web_source)
        self.assertNotIn("web_app", infrastructure_source)
        self.assertNotIn("agent.application", infrastructure_source)

    def test_scene_and_semantic_action_have_one_domain_identity(self) -> None:
        import agent.domain.canonical_action_protocol as canonical_protocol
        from agent.domain.semantic_action import SemanticAction
        from agent.domain.ui_scene import UIScene

        root = Path(__file__).resolve().parent
        self.assertFalse((root / "semantic_action.py").exists())
        self.assertFalse((root / "ui_scene.py").exists())
        self.assertIs(SemanticAction, canonical_protocol.SemanticAction)
        self.assertIs(UIScene, canonical_protocol.UIScene)

        legacy_imports: list[str] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module in {
                        "semantic_action",
                        "ui_scene",
                    }
                ):
                    legacy_imports.append(
                        f"{path.relative_to(root)}: {node.module}"
                    )
                if isinstance(node, ast.Import):
                    for item in node.names:
                        if item.name in {"semantic_action", "ui_scene"}:
                            legacy_imports.append(
                                f"{path.relative_to(root)}: {item.name}"
                            )
        self.assertEqual([], legacy_imports)

        allowed = {"__future__", "dataclasses", "re", "typing"}
        unexpected: list[str] = []
        for path in (
            root / "agent" / "domain" / "semantic_action.py",
            root / "agent" / "domain" / "ui_scene.py",
        ):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [item.name for item in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    if name.split(".")[0] not in allowed:
                        unexpected.append(f"{path.name}: {name}")
        self.assertEqual([], unexpected)

    def test_verified_text_planner_has_one_domain_identity(self) -> None:
        import agent.domain.canonical_action_protocol as canonical_protocol
        from agent.domain.verified_text_transaction import (
            VerifiedTextTransactionError,
        )

        root = Path(__file__).resolve().parent
        self.assertFalse((root / "text_input_utils.py").exists())
        self.assertFalse((root / "verified_text_transaction.py").exists())
        self.assertIs(
            VerifiedTextTransactionError,
            canonical_protocol.VerifiedTextTransactionError,
        )

        legacy_imports: list[str] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module
                    in {"text_input_utils", "verified_text_transaction"}
                ):
                    legacy_imports.append(
                        f"{path.relative_to(root)}: {node.module}"
                    )
                if isinstance(node, ast.Import):
                    for item in node.names:
                        if item.name in {
                            "text_input_utils",
                            "verified_text_transaction",
                        }:
                            legacy_imports.append(
                                f"{path.relative_to(root)}: {item.name}"
                            )
        self.assertEqual([], legacy_imports)

        allowed = {
            "__future__",
            "dataclasses",
            "pypinyin",
            "re",
            "typing",
            "unicodedata",
        }
        unexpected: list[str] = []
        for path in (
            root / "agent" / "domain" / "text_input_utils.py",
            root / "agent" / "domain" / "verified_text_transaction.py",
        ):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [item.name for item in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    if isinstance(node, ast.ImportFrom) and node.level:
                        continue
                    if name.split(".")[0] not in allowed:
                        unexpected.append(f"{path.name}: {name}")
        self.assertEqual([], unexpected)

    def test_typed_input_lineage_has_one_layered_runtime_entry(self) -> None:
        from agent.application.input_value_lineage import (
            TypedInputLineageStorePort,
        )
        from agent.domain.input_value_lineage import TypedInputLineage
        from agent.infrastructure.file_system_input_lineage_store import (
            FileSystemTypedInputLineageStore,
        )

        root = Path(__file__).resolve().parent
        domain_path = root / "agent" / "domain" / "input_value_lineage.py"
        application_path = (
            root / "agent" / "application" / "input_value_lineage.py"
        )
        infrastructure_path = (
            root
            / "agent"
            / "infrastructure"
            / "file_system_input_lineage_store.py"
        )
        self.assertFalse((root / "input_value_lineage.py").exists())
        self.assertTrue(domain_path.is_file())
        self.assertTrue(application_path.is_file())
        self.assertTrue(infrastructure_path.is_file())
        self.assertTrue(issubclass(FileSystemTypedInputLineageStore, object))
        self.assertTrue(hasattr(TypedInputLineage, "matches_typed_context"))
        self.assertTrue(hasattr(TypedInputLineageStorePort, "load"))

        domain_source = domain_path.read_text(encoding="utf-8")
        application_source = application_path.read_text(encoding="utf-8")
        infrastructure_source = infrastructure_path.read_text(encoding="utf-8")
        for forbidden in (
            "from PIL",
            "import PIL",
            "from pathlib",
            "import os",
            "import tempfile",
            "agent.application",
            "agent.infrastructure",
        ):
            self.assertNotIn(forbidden, domain_source)
        self.assertNotIn("agent.infrastructure", application_source)
        self.assertNotIn("class TypedInputLineageStore", domain_source)
        self.assertEqual(
            1,
            infrastructure_source.count("class FileSystemTypedInputLineageStore"),
        )

        legacy_imports: list[str] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module == "input_value_lineage"
                ):
                    legacy_imports.append(str(path.relative_to(root)))
                if isinstance(node, ast.Import):
                    for item in node.names:
                        if item.name == "input_value_lineage":
                            legacy_imports.append(str(path.relative_to(root)))
        self.assertEqual([], legacy_imports)

    def test_action_capabilities_have_one_domain_identity(self) -> None:
        import runtime_doctor
        from agent.domain.action_capabilities import (
            KNOWN_ACTION_CAPABILITIES,
            build_device_capability_snapshot,
        )

        root = Path(__file__).resolve().parent
        domain_path = root / "agent" / "domain" / "action_capabilities.py"
        self.assertFalse((root / "action_capabilities.py").exists())
        self.assertTrue(domain_path.is_file())
        self.assertIs(
            build_device_capability_snapshot,
            runtime_doctor.build_device_capability_snapshot,
        )
        self.assertIn("tap_semantic", KNOWN_ACTION_CAPABILITIES)

        allowed = {
            "__future__",
            "dataclasses",
            "hashlib",
            "json",
            "re",
            "typing",
        }
        unexpected: list[str] = []
        tree = ast.parse(domain_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.split(".")[0] not in allowed:
                    unexpected.append(name)
        self.assertEqual([], unexpected)

        legacy_imports: list[str] = []
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module == "action_capabilities"
                ):
                    legacy_imports.append(str(path.relative_to(root)))
                if not isinstance(node, ast.Import):
                    continue
                for item in node.names:
                    if item.name == "action_capabilities":
                        legacy_imports.append(str(path.relative_to(root)))
        self.assertEqual([], legacy_imports)

    def test_canonical_action_kinds_have_one_domain_identity(self) -> None:
        import agent.domain.canonical_action_protocol as canonical_protocol
        import qwen_visual_decision
        from agent.domain.action_capabilities import KNOWN_ACTION_CAPABILITIES
        from agent.domain.canonical_action_kinds import CANONICAL_ACTION_KINDS
        from agent.domain.device_execution import EXECUTABLE_ACTION_KINDS

        expected = {
            "back",
            "clear_verified_text",
            "dismiss_overlay",
            "double_tap",
            "drag",
            "home",
            "input_verified_text",
            "long_press",
            "open_recent_apps",
            "press_enter",
            "reveal_system_navigation",
            "swipe",
            "tap_semantic",
            "wait_for_change",
        }
        self.assertEqual(expected, CANONICAL_ACTION_KINDS)
        self.assertIs(
            CANONICAL_ACTION_KINDS,
            canonical_protocol.CANONICAL_ACTION_KINDS,
        )
        self.assertIs(CANONICAL_ACTION_KINDS, EXECUTABLE_ACTION_KINDS)
        self.assertEqual(
            CANONICAL_ACTION_KINDS,
            qwen_visual_decision.QWEN_PROTOCOL_ACTIONS,
        )
        self.assertEqual(
            {"hardware_key", "pinch"},
            KNOWN_ACTION_CAPABILITIES - CANONICAL_ACTION_KINDS,
        )

        root = Path(__file__).resolve().parent
        canonical_source = (
            root / "agent" / "domain" / "canonical_action_protocol.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("SUPPORTED_ACTIONS =", canonical_source)
        for path in (
            root / "agent" / "domain" / "device_execution.py",
            root / "agent" / "application" / "runtime_session.py",
        ):
            self.assertNotIn(
                "canonical_action_protocol",
                path.read_text(encoding="utf-8"),
            )

    def test_task_semantic_ir_has_one_domain_identity_and_one_loader(self) -> None:
        import agent.domain.canonical_action_protocol as canonical_protocol
        import agent.application.deepseek_task_graph as deepseek_task_graph
        from agent.domain.task_semantic_ir import (
            TaskSemanticIR,
            compile_formal_semantic_authority,
        )

        root = Path(__file__).resolve().parent
        domain_path = root / "agent" / "domain" / "task_semantic_ir.py"
        loader_path = (
            root
            / "agent"
            / "infrastructure"
            / "file_system_risk_policy.py"
        )
        self.assertFalse((root / "task_semantic_ir.py").exists())
        self.assertTrue(domain_path.is_file())
        self.assertTrue(loader_path.is_file())
        self.assertIs(TaskSemanticIR, canonical_protocol.TaskSemanticIR)
        self.assertIs(
            compile_formal_semantic_authority,
            deepseek_task_graph.compile_formal_semantic_authority,
        )

        domain_source = domain_path.read_text(encoding="utf-8")
        loader_source = loader_path.read_text(encoding="utf-8")
        for forbidden in (
            "from pathlib",
            "Path(",
            ".read_text(",
            "def load_local_risk_policy(",
            "agent.infrastructure",
        ):
            self.assertNotIn(forbidden, domain_source)
        self.assertEqual(1, loader_source.count("def load_local_risk_policy("))
        self.assertEqual(1, loader_source.count(".read_text("))

        legacy_imports: list[str] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module == "task_semantic_ir"
                ):
                    legacy_imports.append(str(path.relative_to(root)))
                if isinstance(node, ast.Import):
                    for item in node.names:
                        if item.name == "task_semantic_ir":
                            legacy_imports.append(str(path.relative_to(root)))
        self.assertEqual([], legacy_imports)

    def test_canonical_action_catalog_has_one_domain_identity(self) -> None:
        import qwen_visual_decision
        from agent.domain.canonical_action_protocol import (
            CanonicalActionCandidate,
            GenericStepProposal,
        )

        root = Path(__file__).resolve().parent
        domain_path = (
            root / "agent" / "domain" / "canonical_action_protocol.py"
        )
        self.assertFalse((root / "canonical_action_protocol.py").exists())
        self.assertTrue(domain_path.is_file())
        self.assertIs(GenericStepProposal, qwen_visual_decision.GenericStepProposal)
        self.assertTrue(hasattr(CanonicalActionCandidate, "to_dict"))

        source = domain_path.read_text(encoding="utf-8")
        for forbidden in (
            "fastapi",
            "pydantic",
            "web_app",
            "agent.application",
            "agent.infrastructure",
            "vision_agent",
            "robot_core",
        ):
            self.assertNotIn(forbidden, source)

        legacy_imports: list[str] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module == "canonical_action_protocol"
                ):
                    legacy_imports.append(str(path.relative_to(root)))
                if isinstance(node, ast.Import):
                    for item in node.names:
                        if item.name == "canonical_action_protocol":
                            legacy_imports.append(str(path.relative_to(root)))
        self.assertEqual([], legacy_imports)

    def test_generic_goal_projection_has_one_domain_identity(self) -> None:
        import agent.application.deepseek_task_graph as deepseek_task_graph
        import generic_action_adapter
        import universal_agent_orchestrator
        from agent.application import runtime_session
        from agent.domain.generic_goal import (
            GenericIntentDraft,
            _parse_json_object,
        )

        root = Path(__file__).resolve().parent
        domain_path = root / "agent" / "domain" / "generic_goal.py"
        self.assertFalse((root / "generic_goal.py").exists())
        self.assertTrue(domain_path.is_file())
        self.assertIs(GenericIntentDraft, generic_action_adapter.GenericIntentDraft)
        self.assertIs(GenericIntentDraft, runtime_session.GenericIntentDraft)
        self.assertIs(GenericIntentDraft, universal_agent_orchestrator.GenericIntentDraft)
        self.assertIs(_parse_json_object, deepseek_task_graph._parse_json_object)

        source = domain_path.read_text(encoding="utf-8")
        for forbidden in (
            "fastapi",
            "pydantic",
            "web_app",
            "agent.application",
            "agent.infrastructure",
            "vision_agent",
            "robot_core",
        ):
            self.assertNotIn(forbidden, source)

        legacy_imports: list[str] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module == "generic_goal"
                ):
                    legacy_imports.append(str(path.relative_to(root)))
                if isinstance(node, ast.Import):
                    for item in node.names:
                        if item.name == "generic_goal":
                            legacy_imports.append(str(path.relative_to(root)))
        self.assertEqual([], legacy_imports)

    def test_deepseek_task_graph_has_one_application_entry(self) -> None:
        import agent.application.deepseek_task_graph as deepseek_task_graph
        import capability_acceptance_planner
        import universal_agent_orchestrator
        from agent.domain.task_graph import DynamicTaskGraph, ObservedState

        root = Path(__file__).resolve().parent
        application_path = (
            root / "agent" / "application" / "deepseek_task_graph.py"
        )
        self.assertFalse((root / "deepseek_task_graph.py").exists())
        self.assertTrue(application_path.is_file())
        self.assertIs(
            DynamicTaskGraph,
            universal_agent_orchestrator.DynamicTaskGraph,
        )
        self.assertIs(
            ObservedState,
            capability_acceptance_planner.ObservedState,
        )
        self.assertEqual(
            ("DeepSeekTaskGraphPlanner", "JsonTaskGraphProvider"),
            deepseek_task_graph.__all__,
        )

        application_source = application_path.read_text(encoding="utf-8")
        for forbidden in (
            "agent.infrastructure",
            "from pathlib",
            "Path(",
            ".read_text(",
            "load_local_risk_policy",
            "fastapi",
            "pydantic",
            "web_app",
            "vision_agent",
            "robot_core",
        ):
            self.assertNotIn(forbidden, application_source)
        self.assertIn("else LocalRiskPolicyConfig()", application_source)
        self.assertNotIn("@dataclass", application_source)
        self.assertNotIn("class DynamicTaskGraph", application_source)
        self.assertNotIn("def _graph_from_payload", application_source)

        web_source = (root / "web_app.py").read_text(encoding="utf-8")
        self.assertIn(
            "from agent.infrastructure.file_system_risk_policy import "
            "load_local_risk_policy",
            web_source,
        )
        self.assertIn("semantic_risk_policy=load_local_risk_policy(", web_source)

        legacy_imports: list[str] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module == "deepseek_task_graph"
                ):
                    legacy_imports.append(str(path.relative_to(root)))
                if isinstance(node, ast.Import):
                    for item in node.names:
                        if item.name == "deepseek_task_graph":
                            legacy_imports.append(str(path.relative_to(root)))
        self.assertEqual([], legacy_imports)

    def test_task_graph_aggregate_has_one_domain_identity(self) -> None:
        import capability_acceptance_planner
        import universal_agent_orchestrator
        from agent.domain.task_graph import (
            DynamicTaskGraph,
            ObservedState,
            TaskGraphError,
            _graph_from_payload,
        )

        root = Path(__file__).resolve().parent
        domain_path = root / "agent" / "domain" / "task_graph.py"
        self.assertTrue(domain_path.is_file())
        self.assertIs(DynamicTaskGraph, universal_agent_orchestrator.DynamicTaskGraph)
        self.assertIs(ObservedState, capability_acceptance_planner.ObservedState)
        self.assertTrue(issubclass(TaskGraphError, ValueError))
        self.assertTrue(callable(_graph_from_payload))

        domain_source = domain_path.read_text(encoding="utf-8")
        for forbidden in (
            "agent.application",
            "agent.infrastructure",
            "fastapi",
            "pydantic",
            "web_app",
            "vision_agent",
            "robot_core",
            "class DeepSeekTaskGraphPlanner",
            "class JsonTaskGraphProvider",
            "def _initial_prompt",
            "def _replan_prompt",
            "def _schema_prompt",
            "chat_json",
        ):
            self.assertNotIn(forbidden, domain_source)

        misplaced_imports: list[str] = []
        allowed_application_exports = {
            "DeepSeekTaskGraphPlanner",
            "JsonTaskGraphProvider",
        }
        for path in root.rglob("*.py"):
            if path.name.startswith("test_"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module
                    == "agent.application.deepseek_task_graph"
                ):
                    names = {item.name for item in node.names}
                    if not names.issubset(allowed_application_exports):
                        misplaced_imports.append(
                            f"{path.relative_to(root)}: {sorted(names)}"
                        )
        self.assertEqual([], misplaced_imports)

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
