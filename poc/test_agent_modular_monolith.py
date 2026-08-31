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
)
from agent.infrastructure import InMemoryAgentSessionRepository
from agent.domain.validation import bounds_overlap


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
    def test_rectangle_overlap_has_one_domain_authority(self) -> None:
        partial = bounds_overlap((0.0, 0.0, 1.0, 1.0), (0.5, 0.0, 1.5, 1.0))
        contained = bounds_overlap((0, 0, 10, 10), (2, 2, 8, 8))
        disjoint = bounds_overlap((0.0, 0.0, 0.2, 0.2), (0.8, 0.8, 1.0, 1.0))

        self.assertAlmostEqual(1 / 3, partial["iou"])
        self.assertAlmostEqual(0.5, partial["intersection_over_smaller"])
        self.assertAlmostEqual(0.36, contained["iou"])
        self.assertEqual(1.0, contained["intersection_over_smaller"])
        self.assertEqual({"iou": 0.0, "intersection_over_smaller": 0.0}, disjoint)

        root = Path(__file__).resolve().parent / "agent"
        owners: list[str] = []
        legacy_iou_owners: list[str] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                relative = path.relative_to(root).as_posix()
                if node.name == "bounds_overlap":
                    owners.append(relative)
                elif node.name == "_bounds_iou":
                    legacy_iou_owners.append(relative)
        self.assertEqual(["domain/validation.py"], owners)
        self.assertEqual([], legacy_iou_owners)

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
        orchestrator_source = (
            root / "agent" / "application" / "universal_agent_orchestrator.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("class UniversalAgentSessionState", orchestrator_source)
        self.assertNotIn(
            '"2026-08-16-universal-post-action-transition-v1"',
            orchestrator_source,
        )
        self.assertNotIn(
            '"2026-08-24-fresh-observation-corrective-retry-v1"',
            orchestrator_source,
        )

    def test_app_surface_lineage_authority_is_removed_but_snapshot_shape_is_stable(self) -> None:
        root = Path(__file__).resolve().parent
        session_source = (
            root / "agent" / "application" / "runtime_session.py"
        ).read_text(encoding="utf-8")
        self.assertFalse((root / "agent" / "domain" / "app_surface_lineage.py").exists())
        self.assertNotIn("VerifiedAppSurfaceLineage", session_source)
        self.assertIn("'verified_app_surface_lineage': None", session_source)

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
        orchestrator_source = (
            root / "agent" / "application" / "universal_agent_orchestrator.py"
        ).read_text(encoding="utf-8")
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
        orchestrator_source = (
            root / "agent" / "application" / "universal_agent_orchestrator.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("class CanonicalSelectionReceipt", orchestrator_source)
        self.assertNotIn(
            '"2026-08-26-canonical-selection-receipt-v1"',
            orchestrator_source,
        )
        self.assertIn("CanonicalSelectionReceipt(", orchestrator_source)
        self.assertNotIn("_deterministic_exact_selection_payload", orchestrator_source)

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

    def test_orchestrator_has_one_current_observation_decision_stage(self) -> None:
        root = Path(__file__).resolve().parent
        orchestrator_path = (
            root / "agent" / "application" / "universal_agent_orchestrator.py"
        )
        tree = ast.parse(orchestrator_path.read_text(encoding="utf-8"))
        orchestrator = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "UniversalAgentOrchestrator"
        )

        owners: dict[str, list[str]] = {
            "_decide": [],
            "write_qwen_decision": [],
            "_selection_receipt": [],
            "trusted_observation_factory": [],
        }
        for method in orchestrator.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(method):
                if not isinstance(node, ast.Call):
                    continue
                name = (
                    node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else node.func.id
                    if isinstance(node.func, ast.Name)
                    else ""
                )
                if name in owners:
                    owners[name].append(method.name)

        self.assertEqual(
            ["_observe_and_decide", "_confirm_one_locked"],
            owners["_decide"],
        )
        self.assertEqual(
            ["_stage_decision"],
            owners["write_qwen_decision"],
        )
        self.assertEqual(
            ["_stage_decision"],
            owners["_selection_receipt"],
        )
        self.assertEqual(
            ["_build_observation"],
            owners["trusted_observation_factory"],
        )

    def test_universal_orchestrator_and_usage_have_one_layered_entry(self) -> None:
        import agent.application.universal_agent_orchestrator as orchestrator
        import agent.application.vision_usage as vision_usage
        import agent.infrastructure.deepseek_failure_diagnostics as diagnostics

        root = Path(__file__).resolve().parent
        orchestrator_path = (
            root / "agent" / "application" / "universal_agent_orchestrator.py"
        )
        usage_path = root / "agent" / "application" / "vision_usage.py"
        diagnostic_path = (
            root
            / "agent"
            / "infrastructure"
            / "deepseek_failure_diagnostics.py"
        )
        for legacy_name in (
            "universal_agent_orchestrator.py",
            "vision_usage.py",
            "deepseek_failure_diagnostics.py",
        ):
            self.assertFalse((root / legacy_name).exists())
        self.assertTrue(orchestrator_path.is_file())
        self.assertTrue(usage_path.is_file())
        self.assertTrue(diagnostic_path.is_file())
        self.assertEqual(
            "agent.application.universal_agent_orchestrator",
            orchestrator.UniversalAgentOrchestrator.__module__,
        )
        self.assertEqual(
            "agent.application.vision_usage",
            vision_usage.VisionSessionUsageLedger.__module__,
        )
        self.assertEqual(
            "agent.infrastructure.deepseek_failure_diagnostics",
            diagnostics.persist_deepseek_failure_diagnostic.__module__,
        )

        orchestrator_source = orchestrator_path.read_text(encoding="utf-8")
        usage_source = usage_path.read_text(encoding="utf-8")
        diagnostic_source = diagnostic_path.read_text(encoding="utf-8")
        for forbidden in (
            "agent.infrastructure",
            "fastapi",
            "pydantic",
            "web_app",
            "robot_core",
            "vision_agent",
            ".read_text(",
            ".write_text(",
            ".write_bytes(",
            "os.replace",
            ".mkdir(",
        ):
            self.assertNotIn(forbidden, orchestrator_source)
            self.assertNotIn(forbidden, usage_source)
        self.assertIn("deepseek_failure_diagnostic_writer", orchestrator_source)
        self.assertNotIn(
            "persist_deepseek_failure_diagnostic",
            orchestrator_source,
        )
        self.assertEqual(
            1,
            orchestrator_source.count("class UniversalAgentOrchestrator:"),
        )
        self.assertEqual(
            1,
            usage_source.count("class VisionSessionUsageLedger:"),
        )
        self.assertEqual(
            1,
            diagnostic_source.count(
                "def persist_deepseek_failure_diagnostic("
            ),
        )

        web_source = (root / "web_app.py").read_text(encoding="utf-8")
        self.assertIn(
            "from agent.application.universal_agent_orchestrator import (",
            web_source,
        )
        self.assertIn(
            "from agent.infrastructure.deepseek_failure_diagnostics import (",
            web_source,
        )
        self.assertEqual(
            2,
            web_source.count(
                "deepseek_failure_diagnostic_writer=("
            ),
        )

        legacy_modules = {
            "universal_agent_orchestrator",
            "vision_usage",
            "deepseek_failure_diagnostics",
        }
        legacy_imports: list[str] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module in legacy_modules
                ):
                    legacy_imports.append(str(path.relative_to(root)))
                if isinstance(node, ast.Import):
                    for item in node.names:
                        if item.name in legacy_modules:
                            legacy_imports.append(str(path.relative_to(root)))
        self.assertEqual([], legacy_imports)

    def test_deepseek_intent_provider_has_one_infrastructure_entry(self) -> None:
        import agent.infrastructure.deepseek_intent_provider as provider

        root = Path(__file__).resolve().parent
        provider_path = (
            root / "agent" / "infrastructure" / "deepseek_intent_provider.py"
        )
        self.assertFalse((root / "intent_provider.py").exists())
        self.assertTrue(provider_path.is_file())
        self.assertEqual(
            "agent.infrastructure.deepseek_intent_provider",
            provider.DeepSeekIntentProvider.__module__,
        )
        source = provider_path.read_text(encoding="utf-8")
        self.assertEqual(1, source.count("class DeepSeekIntentProvider:"))
        self.assertEqual(1, source.count("class IntentProviderError("))
        for forbidden in (
            "fastapi",
            "pydantic",
            "web_app",
            "robot_core",
            "vision_agent",
        ):
            self.assertNotIn(forbidden, source)

        web_source = (root / "web_app.py").read_text(encoding="utf-8")
        self.assertIn(
            "from agent.infrastructure.deepseek_intent_provider import (",
            web_source,
        )
        legacy_imports: list[str] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module == "intent_provider"
                ):
                    legacy_imports.append(str(path.relative_to(root)))
                if isinstance(node, ast.Import):
                    for item in node.names:
                        if item.name == "intent_provider":
                            legacy_imports.append(str(path.relative_to(root)))
        self.assertEqual([], legacy_imports)

    def test_windows_ocr_runtime_has_one_infrastructure_entry(self) -> None:
        import agent.infrastructure.windows_ocr_runtime as ocr_runtime

        root = Path(__file__).resolve().parent
        runtime_path = (
            root / "agent" / "infrastructure" / "windows_ocr_runtime.py"
        )
        self.assertFalse((root / "ocr_runtime.py").exists())
        self.assertTrue(runtime_path.is_file())
        self.assertEqual(root / "windows_ocr.ps1", ocr_runtime.OCR_SCRIPT)
        self.assertEqual(
            "agent.infrastructure.windows_ocr_runtime",
            ocr_runtime.OcrMatch.__module__,
        )
        payload = {
            "lines": [
                {
                    "text": "文件 传输 助手",
                    "words": [
                        {"text": "文件", "left": 10, "top": 20, "width": 30, "height": 12},
                        {"text": "传输", "left": 42, "top": 20, "width": 30, "height": 12},
                        {"text": "助手", "left": 74, "top": 20, "width": 30, "height": 12},
                    ],
                }
            ]
        }
        matches = ocr_runtime.find_text(payload, "传输助手")
        self.assertEqual(1, len(matches))
        self.assertEqual((73, 26), matches[0].center)
        self.assertEqual([], ocr_runtime.find_text(payload, "   "))

        source = runtime_path.read_text(encoding="utf-8")
        self.assertEqual(1, source.count("class OcrMatch:"))
        self.assertEqual(1, source.count("def recognize("))
        self.assertIn("Path(__file__).resolve().parents[2]", source)
        legacy_imports: list[str] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module == "ocr_runtime"
                ):
                    legacy_imports.append(str(path.relative_to(root)))
                if isinstance(node, ast.Import):
                    for item in node.names:
                        if item.name == "ocr_runtime":
                            legacy_imports.append(str(path.relative_to(root)))
        self.assertEqual([], legacy_imports)

    def test_orientation_safety_has_one_infrastructure_entry(self) -> None:
        import agent.infrastructure.orientation_safety as orientation_safety

        root = Path(__file__).resolve().parent
        runtime_path = (
            root / "agent" / "infrastructure" / "orientation_safety.py"
        )
        self.assertFalse((root / "orientation_safety.py").exists())
        self.assertTrue(runtime_path.is_file())
        self.assertEqual(
            "agent.infrastructure.orientation_safety",
            orientation_safety.OrientationCredential.__module__,
        )
        self.assertEqual(
            "agent.infrastructure.orientation_safety",
            orientation_safety.PhysicalExecutionGate.__module__,
        )
        source = runtime_path.read_text(encoding="utf-8")
        self.assertEqual(1, source.count("class OrientationCredential:"))
        self.assertEqual(1, source.count("class PhysicalExecutionGate:"))
        for forbidden in ("fastapi", "pydantic", "web_app", "robot_core"):
            self.assertNotIn(forbidden, source)

        legacy_imports: list[str] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module == "orientation_safety"
                ):
                    legacy_imports.append(str(path.relative_to(root)))
                if isinstance(node, ast.Import):
                    for item in node.names:
                        if item.name == "orientation_safety":
                            legacy_imports.append(str(path.relative_to(root)))
        self.assertEqual([], legacy_imports)

    def test_tap_calibration_has_one_infrastructure_entry(self) -> None:
        import agent.infrastructure.tap_calibration as tap_calibration

        root = Path(__file__).resolve().parent
        runtime_path = (
            root / "agent" / "infrastructure" / "tap_calibration.py"
        )
        self.assertFalse((root / "tap_calibration.py").exists())
        self.assertTrue(runtime_path.is_file())
        self.assertEqual(
            root / "tap_calibration.json",
            tap_calibration.CALIBRATION_PATH,
        )
        self.assertEqual(
            "agent.infrastructure.tap_calibration",
            tap_calibration.Affine2D.__module__,
        )
        coverage = tap_calibration.build_coverage(
            [(0.05, 0.04), (0.95, 0.04), (0.95, 0.96), (0.05, 0.96)]
        )
        self.assertTrue(coverage["sufficient"])
        self.assertEqual([0.9, 0.92], coverage["span"])

        source = runtime_path.read_text(encoding="utf-8")
        self.assertEqual(1, source.count("class Affine2D:"))
        self.assertEqual(1, source.count("class TapCalibrationError("))
        self.assertEqual(1, source.count("def corrected_grid_point("))
        self.assertIn("Path(__file__).resolve().parents[2]", source)
        for forbidden in ("fastapi", "pydantic", "web_app", "robot_core"):
            self.assertNotIn(forbidden, source)

        legacy_imports: list[str] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module == "tap_calibration"
                ):
                    legacy_imports.append(str(path.relative_to(root)))
                if isinstance(node, ast.Import):
                    for item in node.names:
                        if item.name == "tap_calibration":
                            legacy_imports.append(str(path.relative_to(root)))
        self.assertEqual([], legacy_imports)

    def test_seller_window_adapter_has_one_infrastructure_entry(self) -> None:
        import agent.infrastructure.seller_window_adapter as seller_window

        root = Path(__file__).resolve().parent
        adapter_path = (
            root / "agent" / "infrastructure" / "seller_window_adapter.py"
        )
        self.assertFalse((root / "robot_gui_poc.py").exists())
        self.assertTrue(adapter_path.is_file())
        self.assertEqual(root, seller_window.ROOT)
        self.assertEqual(root / "output", seller_window.OUTPUT_DIR)
        self.assertEqual(
            "agent.infrastructure.seller_window_adapter",
            seller_window.INPUT.__module__,
        )
        self.assertEqual(1.25, seller_window.seller_ui_scale(675))
        self.assertEqual(
            (2557, 2),
            seller_window.cursor_parking_screen_point(
                (0, 0, 830, 1600),
                (0, 0, 2560, 1600),
            ),
        )

        source = adapter_path.read_text(encoding="utf-8")
        self.assertEqual(1, source.count("class INPUT(ctypes.Structure):"))
        self.assertEqual(1, source.count("def capture_client_passive("))
        self.assertEqual(1, source.count("def temporarily_park_cursor_outside_camera("))
        self.assertIn("Path(__file__).resolve().parents[2]", source)
        for forbidden in (
            "fastapi",
            "pydantic",
            "web_app",
            "robot_core",
            "canonical_action",
        ):
            self.assertNotIn(forbidden, source)

        legacy_imports: list[str] = []
        legacy_patches: list[str] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module == "robot_gui_poc"
                ):
                    legacy_imports.append(str(path.relative_to(root)))
                if isinstance(node, ast.Import):
                    for item in node.names:
                        if item.name == "robot_gui_poc":
                            legacy_imports.append(str(path.relative_to(root)))
                if (
                    path.name != "test_agent_modular_monolith.py"
                    and isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.value.startswith("robot_gui_poc.")
                ):
                    legacy_patches.append(str(path.relative_to(root)))
        self.assertEqual([], legacy_imports)
        self.assertEqual([], legacy_patches)

    def test_robot_controller_has_one_infrastructure_entry(self) -> None:
        import agent.infrastructure.robot_controller as robot_controller

        root = Path(__file__).resolve().parent
        controller_path = (
            root / "agent" / "infrastructure" / "robot_controller.py"
        )
        self.assertFalse((root / "robot_core.py").exists())
        self.assertTrue(controller_path.is_file())
        self.assertEqual(root, robot_controller.POC_ROOT)
        self.assertEqual(
            root / "controller_config.json",
            robot_controller.CONTROL_CONFIG_PATH,
        )
        self.assertEqual(root / "output" / "web", robot_controller.WEB_OUTPUT_DIR)
        self.assertEqual(
            "agent.infrastructure.robot_controller",
            robot_controller.RobotController.__module__,
        )
        self.assertEqual(
            "agent.infrastructure.robot_controller",
            robot_controller.MockRobotController.__module__,
        )
        mock = robot_controller.MockRobotController(device_id="architecture-test")
        self.assertEqual(root / "tap_calibration.json", mock.calibration_path)
        mock.request_stop()
        self.assertTrue(mock.stop_event.is_set())
        mock.begin_new_task()
        self.assertFalse(mock.stop_event.is_set())

        source = controller_path.read_text(encoding="utf-8")
        self.assertEqual(1, source.count("class RobotController:"))
        self.assertEqual(1, source.count("class MockRobotController(RobotController):"))
        self.assertEqual(1, source.count("def load_controller_config("))
        self.assertEqual(1, source.count("def qwerty_keyboard_config_from_anchors("))
        self.assertIn("Path(__file__).resolve().parents[2]", source)
        for forbidden in (
            "agent.application",
            "fastapi",
            "pydantic",
            "web_app",
            "capability_acceptance",
        ):
            self.assertNotIn(forbidden, source)

        legacy_imports: list[str] = []
        legacy_patches: list[str] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module == "robot_core"
                ):
                    legacy_imports.append(str(path.relative_to(root)))
                if isinstance(node, ast.Import):
                    for item in node.names:
                        if item.name == "robot_core":
                            legacy_imports.append(str(path.relative_to(root)))
                if (
                    path.name != "test_agent_modular_monolith.py"
                    and isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.value.startswith("robot_core.")
                ):
                    legacy_patches.append(str(path.relative_to(root)))
        self.assertEqual([], legacy_imports)
        self.assertEqual([], legacy_patches)

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

        allowed = {"__future__", "dataclasses", "re", "typing", "validation"}
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

    def test_root_production_modules_are_only_interfaces_and_tools(self) -> None:
        root = Path(__file__).resolve().parent
        expected = {
            "agent_api_cli.py",
            "capture_click_burst.py",
            "companion_ime_setup.py",
            "eval_qwen_visual_decision.py",
            "eval_task_sequences.py",
            "local_agent_api_client.py",
            "run_xy_calibration.py",
            "touch_calibration_server.py",
            "web_app.py",
        }
        actual = {
            path.name
            for path in root.glob("*.py")
            if not path.name.startswith("test_")
        }
        self.assertEqual(expected, actual)

        web_source = (root / "web_app.py").read_text(encoding="utf-8")
        client_source = (root / "local_agent_api_client.py").read_text(
            encoding="utf-8"
        )
        cli_source = (root / "agent_api_cli.py").read_text(encoding="utf-8")
        self.assertIn("from fastapi import", web_source)
        self.assertIn('"/openapi.json"', client_source)
        self.assertIn("LocalAgentApiClient", cli_source)

        root_module_names = {path.removesuffix(".py") for path in expected}
        reverse_imports: list[str] = []
        for path in (root / "agent").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module
                ):
                    names = [node.module]
                for name in names:
                    if name.split(".", 1)[0] in root_module_names:
                        reverse_imports.append(
                            f"{path.relative_to(root)}: {name}"
                        )
        self.assertEqual([], reverse_imports)

    def test_runtime_doctor_has_one_infrastructure_entry(self) -> None:
        import agent.infrastructure.runtime_doctor as runtime_doctor
        from agent.application.vision_usage import QWEN_PLUS_MODEL
        from agent.domain.vision_model import DEFAULT_VISION_MODEL

        root = Path(__file__).resolve().parent
        infrastructure_path = (
            root / "agent" / "infrastructure" / "runtime_doctor.py"
        )
        self.assertFalse((root / "runtime_doctor.py").exists())
        self.assertTrue(infrastructure_path.is_file())
        self.assertIs(QWEN_PLUS_MODEL, DEFAULT_VISION_MODEL)
        self.assertIs(
            runtime_doctor.DEFAULT_VISION_MODEL,
            DEFAULT_VISION_MODEL,
        )
        self.assertEqual(
            "agent.infrastructure.runtime_doctor",
            runtime_doctor.run_runtime_doctor.__module__,
        )

        source = infrastructure_path.read_text(encoding="utf-8")
        self.assertEqual(1, source.count("def run_runtime_doctor("))
        self.assertEqual(1, source.count("def _capture_stable_frames("))
        for forbidden in (
            "agent.application",
            "fastapi",
            "pydantic",
            "web_app",
            "from runtime_doctor",
        ):
            self.assertNotIn(forbidden, source)

        legacy_imports: list[str] = []
        legacy_patches: list[str] = []
        model_literal_authorities: list[str] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module == "runtime_doctor"
                ):
                    legacy_imports.append(str(path.relative_to(root)))
                if isinstance(node, ast.Import):
                    for item in node.names:
                        if item.name == "runtime_doctor":
                            legacy_imports.append(str(path.relative_to(root)))
                if (
                    path.name != "test_agent_modular_monolith.py"
                    and isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.value.startswith("runtime_doctor.")
                ):
                    legacy_patches.append(str(path.relative_to(root)))
                if (
                    not path.name.startswith("test_")
                    and isinstance(node, (ast.Assign, ast.AnnAssign))
                ):
                    targets = (
                        node.targets
                        if isinstance(node, ast.Assign)
                        else [node.target]
                    )
                    value = node.value
                    if (
                        any(
                            isinstance(target, ast.Name)
                            and target.id
                            in {"DEFAULT_VISION_MODEL", "QWEN_PLUS_MODEL"}
                            for target in targets
                        )
                        and isinstance(value, ast.Constant)
                        and value.value == "qwen3.7-plus"
                    ):
                        model_literal_authorities.append(str(path.relative_to(root)))
        self.assertEqual([], legacy_imports)
        self.assertEqual([], legacy_patches)
        self.assertEqual(
            ["agent\\domain\\vision_model.py"],
            model_literal_authorities,
        )

    def test_action_capabilities_have_one_domain_identity(self) -> None:
        import agent.infrastructure.runtime_doctor as runtime_doctor
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
            "validation",
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
        import agent.application.qwen_visual_decision as qwen_visual_decision
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
            "launch_app",
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

    def test_task_semantic_ir_has_one_domain_identity_and_fixed_risk_boundary(self) -> None:
        import agent.domain.canonical_action_protocol as canonical_protocol
        import agent.application.deepseek_task_graph as deepseek_task_graph
        from agent.domain.task_semantic_ir import (
            TaskSemanticIR,
            compile_formal_semantic_authority,
        )

        root = Path(__file__).resolve().parent
        domain_path = root / "agent" / "domain" / "task_semantic_ir.py"
        retired_loader = root / "agent" / "infrastructure" / "file_system_risk_policy.py"
        retired_config = root / "config" / "local_risk_policy.v1.json"
        self.assertFalse((root / "task_semantic_ir.py").exists())
        self.assertTrue(domain_path.is_file())
        self.assertFalse(retired_loader.exists())
        self.assertFalse(retired_config.exists())
        self.assertIs(TaskSemanticIR, canonical_protocol.TaskSemanticIR)
        self.assertIs(
            compile_formal_semantic_authority,
            deepseek_task_graph.compile_formal_semantic_authority,
        )

        domain_source = domain_path.read_text(encoding="utf-8")
        for forbidden in (
            "from pathlib",
            "Path(",
            ".read_text(",
            "def load_local_risk_policy(",
            "LocalRiskPolicyConfig",
            "effect_id_override",
            "agent.infrastructure",
        ):
            self.assertNotIn(forbidden, domain_source)
        self.assertIn('DEFAULT_CONFIRMATION_EFFECT_KINDS = frozenset({"authentication", "financial_transaction"})', domain_source)

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
        import agent.application.qwen_visual_decision as qwen_visual_decision
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
        import agent.infrastructure.generic_action_adapter as generic_action_adapter
        import agent.application.universal_agent_orchestrator as universal_agent_orchestrator
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

    def test_goal_context_semantics_have_one_domain_owner(self) -> None:
        import agent.infrastructure.generic_scene_observer as generic_scene_observer
        import agent.application.qwen_visual_decision as qwen_visual_decision
        from agent.domain.generic_goal import safe_goal_context

        root = Path(__file__).resolve().parent
        generic_goal_path = root / "agent" / "domain" / "generic_goal.py"
        observer_path = (
            root / "agent" / "infrastructure" / "generic_scene_observer.py"
        )
        self.assertFalse((root / "message_intent.py").exists())
        self.assertFalse((root / "agent" / "domain" / "message_intent.py").exists())
        self.assertFalse(hasattr(generic_scene_observer, "_safe_goal_context"))
        self.assertFalse(hasattr(qwen_visual_decision, "safe_goal_context"))
        self.assertFalse(hasattr(qwen_visual_decision, "subgoal_binds_recipient"))

        self.assertEqual(
            {"objective": "选择 Alice", "values": [1, True, None]},
            safe_goal_context(
                {"objective": "选择 Alice", "values": [1, True, None]}
            ),
        )

        generic_goal_source = generic_goal_path.read_text(encoding="utf-8")
        observer_source = observer_path.read_text(encoding="utf-8")
        self.assertEqual(1, generic_goal_source.count("def safe_goal_context("))
        self.assertNotIn("def _safe_goal_context(", observer_source)

    def test_qwen_task_context_has_one_domain_identity(self) -> None:
        import agent.application.qwen_visual_decision as qwen_visual_decision
        import agent.application.universal_agent_orchestrator as universal_agent_orchestrator
        from agent.domain.qwen_task_context import (
            QwenTaskContext,
            SUPPORTED_TASK_CONTEXT_PROTOCOL,
        )

        root = Path(__file__).resolve().parent
        domain_path = root / "agent" / "domain" / "qwen_task_context.py"
        provider_path = (
            root / "agent" / "application" / "qwen_visual_decision.py"
        )
        self.assertTrue(domain_path.is_file())
        self.assertFalse((root / "qwen_task_context.py").exists())
        self.assertFalse(hasattr(qwen_visual_decision, "QwenTaskContext"))
        self.assertIs(QwenTaskContext, universal_agent_orchestrator.QwenTaskContext)
        self.assertEqual(
            "2026-08-20-deepseek-typed-task-graph-v4",
            SUPPORTED_TASK_CONTEXT_PROTOCOL,
        )
        self.assertEqual("agent.domain.qwen_task_context", QwenTaskContext.__module__)

        domain_source = domain_path.read_text(encoding="utf-8")
        provider_source = provider_path.read_text(encoding="utf-8")
        self.assertEqual(1, domain_source.count("class QwenTaskContext("))
        self.assertNotIn("class QwenTaskContext(", provider_source)
        self.assertNotIn("def _require_dict(", provider_source)
        self.assertNotIn("def _text_tuple(", provider_source)
        self.assertNotIn("def _dict_tuple(", provider_source)
        for forbidden in (
            "fastapi",
            "pydantic",
            "web_app",
            "agent.application",
            "agent.infrastructure",
            "from PIL",
            "import os",
            "httpx",
            "vision_agent",
            "generic_scene_observer",
            "robot_core",
        ):
            self.assertNotIn(forbidden, domain_source)

        legacy_imports: list[str] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module == "qwen_visual_decision"
                    and any(item.name == "QwenTaskContext" for item in node.names)
                ):
                    legacy_imports.append(str(path.relative_to(root)))
        self.assertEqual([], legacy_imports)

    def test_qwen_visual_decision_has_one_application_entry(self) -> None:
        import agent.application.qwen_visual_decision as qwen_visual_decision
        from agent.application.qwen_visual_decision import (
            QwenVisualDecisionObserver,
        )

        root = Path(__file__).resolve().parent
        application_path = (
            root / "agent" / "application" / "qwen_visual_decision.py"
        )
        self.assertFalse((root / "qwen_visual_decision.py").exists())
        self.assertTrue(application_path.is_file())
        self.assertEqual(
            "agent.application.qwen_visual_decision",
            QwenVisualDecisionObserver.__module__,
        )
        source = application_path.read_text(encoding="utf-8")
        self.assertEqual(1, source.count("class QwenVisualDecisionObserver:"))
        self.assertEqual(
            "2026-09-01-qwen-same-response-action-finish-v7",
            qwen_visual_decision.QWEN_VISUAL_DECISION_PROTOCOL_VERSION,
        )
        for forbidden in (
            "agent.infrastructure",
            "from pathlib",
            "Path(",
            ".read_text(",
            ".write_text(",
            "import os",
            "os.environ",
            "web_app",
            "robot_core",
            "generic_scene_observer",
        ):
            self.assertNotIn(forbidden, source)

        legacy_imports: list[str] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module == "qwen_visual_decision"
                ):
                    legacy_imports.append(str(path.relative_to(root)))
                if isinstance(node, ast.Import):
                    for item in node.names:
                        if item.name == "qwen_visual_decision":
                            legacy_imports.append(str(path.relative_to(root)))
        self.assertEqual([], legacy_imports)

    def test_universal_action_controller_has_one_domain_entry(self) -> None:
        import agent.domain.universal_action_controller as controller_module
        from agent.domain.universal_action_controller import (
            LocalPointGrounding,
            ResolvedSemanticAction,
            UniversalActionController,
            UniversalActionError,
        )

        root = Path(__file__).resolve().parent
        domain_path = (
            root / "agent" / "domain" / "universal_action_controller.py"
        )
        self.assertFalse((root / "universal_action_controller.py").exists())
        self.assertTrue(domain_path.is_file())
        for symbol in (
            LocalPointGrounding,
            ResolvedSemanticAction,
            UniversalActionController,
            UniversalActionError,
        ):
            self.assertEqual(
                "agent.domain.universal_action_controller",
                symbol.__module__,
            )
        self.assertEqual(
            "2026-08-26-universal-action-v17",
            controller_module.UNIVERSAL_CONTROLLER_PROTOCOL_VERSION,
        )

        source = domain_path.read_text(encoding="utf-8")
        self.assertEqual(1, source.count("class UniversalActionController:"))
        self.assertEqual(1, source.count("class ResolvedSemanticAction:"))
        for forbidden in (
            "agent.application",
            "agent.infrastructure",
            "fastapi",
            "pydantic",
            "from pathlib",
            "Path(",
            ".read_text(",
            ".write_text(",
            "import os",
            "os.environ",
            "from PIL",
            "web_app",
            "vision_agent",
            "generic_scene_observer",
            "generic_action_adapter",
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
                    and node.module == "universal_action_controller"
                ):
                    legacy_imports.append(str(path.relative_to(root)))
                if isinstance(node, ast.Import):
                    for item in node.names:
                        if item.name == "universal_action_controller":
                            legacy_imports.append(str(path.relative_to(root)))
        self.assertEqual([], legacy_imports)

    def test_generic_scene_observer_has_one_infrastructure_entry(self) -> None:
        import agent.infrastructure.generic_scene_observer as observer_module
        from agent.domain.post_action_observation import (
            PostActionVisualContext,
        )
        from agent.infrastructure.generic_scene_observer import (
            SingleStepGenericSceneObserver,
        )

        root = Path(__file__).resolve().parent
        infrastructure_path = (
            root / "agent" / "infrastructure" / "generic_scene_observer.py"
        )
        domain_path = root / "agent" / "domain" / "post_action_observation.py"
        self.assertFalse((root / "generic_scene_observer.py").exists())
        self.assertTrue(infrastructure_path.is_file())
        self.assertTrue(domain_path.is_file())
        self.assertEqual(
            "agent.infrastructure.generic_scene_observer",
            SingleStepGenericSceneObserver.__module__,
        )
        self.assertEqual(
            "agent.domain.post_action_observation",
            PostActionVisualContext.__module__,
        )
        self.assertFalse(hasattr(observer_module, "PostActionVisualContext"))
        self.assertEqual(
            "2026-09-01-single-step-scene-action-finish-v5",
            observer_module.SINGLE_STEP_SCENE_OBSERVER_VERSION,
        )

        source = infrastructure_path.read_text(encoding="utf-8")
        domain_source = domain_path.read_text(encoding="utf-8")
        self.assertEqual(1, source.count("class SingleStepGenericSceneObserver("))
        self.assertNotIn("class PostActionVisualContext:", source)
        self.assertEqual(1, domain_source.count("class PostActionVisualContext("))
        for forbidden in (
            "agent.application",
            "agent.infrastructure",
            "from PIL",
            "fastapi",
            "pydantic",
            "web_app",
            "robot_core",
        ):
            self.assertNotIn(forbidden, domain_source)
        for forbidden in (
            "fastapi",
            "pydantic",
            "web_app",
            "universal_agent_orchestrator",
            "generic_action_adapter",
        ):
            self.assertNotIn(forbidden, source)

        legacy_imports: list[str] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module == "generic_scene_observer"
                ):
                    legacy_imports.append(str(path.relative_to(root)))
                if isinstance(node, ast.Import):
                    for item in node.names:
                        if item.name == "generic_scene_observer":
                            legacy_imports.append(str(path.relative_to(root)))
        self.assertEqual([], legacy_imports)

    def test_generic_action_adapter_has_one_infrastructure_entry(self) -> None:
        import agent.infrastructure.generic_action_adapter as adapter_module
        from agent.application.action_adapter import (
            GenericActionAdapterError,
            GenericSingleActionAdapterPort,
        )
        from agent.infrastructure.generic_action_adapter import (
            GenericActionExecutionResult,
            GenericSingleActionAdapter,
        )

        root = Path(__file__).resolve().parent
        infrastructure_path = (
            root / "agent" / "infrastructure" / "generic_action_adapter.py"
        )
        application_path = root / "agent" / "application" / "action_adapter.py"
        self.assertFalse((root / "generic_action_adapter.py").exists())
        self.assertTrue(infrastructure_path.is_file())
        self.assertTrue(application_path.is_file())
        for symbol in (
            GenericActionExecutionResult,
            GenericSingleActionAdapter,
        ):
            self.assertEqual(
                "agent.infrastructure.generic_action_adapter",
                symbol.__module__,
            )
        self.assertEqual(
            "agent.application.action_adapter",
            GenericActionAdapterError.__module__,
        )
        self.assertEqual(
            "agent.application.action_adapter",
            GenericSingleActionAdapterPort.__module__,
        )
        self.assertEqual(
            "2026-08-17-qwen-failure-diagnostic-v1",
            adapter_module.QWEN_FAILURE_DIAGNOSTIC_VERSION,
        )

        source = infrastructure_path.read_text(encoding="utf-8")
        application_source = application_path.read_text(encoding="utf-8")
        self.assertEqual(1, source.count("class GenericSingleActionAdapter:"))
        self.assertEqual(1, source.count("class GenericActionExecutionResult:"))
        self.assertNotIn("class GenericActionAdapterError(", source)
        self.assertEqual(
            1,
            application_source.count("class GenericActionAdapterError("),
        )
        self.assertEqual(
            1,
            application_source.count("class GenericSingleActionAdapterPort("),
        )
        self.assertNotIn("agent.infrastructure", application_source)
        for forbidden in (
            "fastapi",
            "pydantic",
            "web_app",
            "universal_agent_orchestrator",
        ):
            self.assertNotIn(forbidden, source)

        legacy_imports: list[str] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module == "generic_action_adapter"
                ):
                    legacy_imports.append(str(path.relative_to(root)))
                if isinstance(node, ast.Import):
                    for item in node.names:
                        if item.name == "generic_action_adapter":
                            legacy_imports.append(str(path.relative_to(root)))
        self.assertEqual([], legacy_imports)

    def test_capability_acceptance_has_one_infrastructure_entry(self) -> None:
        import agent.infrastructure.capability_acceptance as acceptance
        from agent.domain.action_capabilities import (
            CALIBRATION_BOUND_ACTIONS,
            PROMOTABLE_ACTIONS,
        )

        root = Path(__file__).resolve().parent
        infrastructure_path = (
            root / "agent" / "infrastructure" / "capability_acceptance.py"
        )
        self.assertFalse((root / "capability_acceptance.py").exists())
        self.assertTrue(infrastructure_path.is_file())
        self.assertIs(PROMOTABLE_ACTIONS, acceptance.PROMOTABLE_ACTIONS)
        self.assertIs(
            CALIBRATION_BOUND_ACTIONS,
            acceptance.CALIBRATION_BOUND_ACTIONS,
        )
        self.assertEqual(
            "agent.infrastructure.capability_acceptance",
            acceptance.CapabilityAcceptanceError.__module__,
        )
        self.assertEqual(
            "agent.infrastructure.capability_acceptance",
            acceptance.PromotionScope.__module__,
        )
        self.assertEqual(
            "agent.infrastructure.capability_acceptance",
            acceptance.PromotionAuthority.__module__,
        )
        self.assertEqual(
            "agent.infrastructure.capability_acceptance",
            acceptance.CapabilityRegistryPromoter.__module__,
        )

        source = infrastructure_path.read_text(encoding="utf-8")
        self.assertEqual(1, source.count("class CapabilityAcceptanceError("))
        self.assertEqual(1, source.count("class PromotionScope("))
        self.assertEqual(1, source.count("class PromotionAuthority:"))
        self.assertEqual(1, source.count("class CapabilityRegistryPromoter:"))
        self.assertEqual(1, source.count("def validate_acceptance_report("))
        for forbidden in (
            "agent.application",
            "fastapi",
            "pydantic",
            "web_app",
            "capability_acceptance_runtime",
        ):
            self.assertNotIn(forbidden, source)

        legacy_imports: list[str] = []
        legacy_patches: list[str] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module == "capability_acceptance"
                ):
                    legacy_imports.append(str(path.relative_to(root)))
                if isinstance(node, ast.Import):
                    for item in node.names:
                        if item.name == "capability_acceptance":
                            legacy_imports.append(str(path.relative_to(root)))
                if (
                    path.name != "test_agent_modular_monolith.py"
                    and isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.value.startswith("capability_acceptance.")
                ):
                    legacy_patches.append(str(path.relative_to(root)))
        self.assertEqual([], legacy_imports)
        self.assertEqual([], legacy_patches)

    def test_capability_acceptance_runtime_has_one_infrastructure_entry(self) -> None:
        import agent.infrastructure.capability_acceptance_runtime as runtime

        root = Path(__file__).resolve().parent
        infrastructure_path = (
            root
            / "agent"
            / "infrastructure"
            / "capability_acceptance_runtime.py"
        )
        self.assertFalse((root / "capability_acceptance_runtime.py").exists())
        self.assertTrue(infrastructure_path.is_file())
        for value in (
            runtime.CapabilityTrial,
            runtime.CapabilityAcceptanceManager,
        ):
            self.assertEqual(
                "agent.infrastructure.capability_acceptance_runtime",
                value.__module__,
            )

        source = infrastructure_path.read_text(encoding="utf-8")
        self.assertEqual(1, source.count("class CapabilityTrial:"))
        self.assertNotIn("class RecoveredCapabilityTrial:", source)
        self.assertEqual(1, source.count("class CapabilityAcceptanceManager:"))
        self.assertEqual(1, source.count("def _atomic_write_json("))
        for forbidden in (
            "agent.application",
            "fastapi",
            "pydantic",
            "web_app",
            "from capability_acceptance_runtime",
        ):
            self.assertNotIn(forbidden, source)

        legacy_imports: list[str] = []
        legacy_patches: list[str] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module == "capability_acceptance_runtime"
                ):
                    legacy_imports.append(str(path.relative_to(root)))
                if isinstance(node, ast.Import):
                    for item in node.names:
                        if item.name == "capability_acceptance_runtime":
                            legacy_imports.append(str(path.relative_to(root)))
                if (
                    path.name != "test_agent_modular_monolith.py"
                    and isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.value.startswith("capability_acceptance_runtime.")
                ):
                    legacy_patches.append(str(path.relative_to(root)))
        self.assertEqual([], legacy_imports)
        self.assertEqual([], legacy_patches)

    def test_capability_acceptance_planner_has_one_application_entry(self) -> None:
        import agent.infrastructure.capability_acceptance as capability_acceptance
        import agent.infrastructure.capability_acceptance_runtime as capability_acceptance_runtime
        import agent.application.capability_acceptance_planner as planner
        from agent.domain.action_capabilities import (
            CALIBRATION_BOUND_ACTIONS,
            PROMOTABLE_ACTIONS,
        )

        root = Path(__file__).resolve().parent
        application_path = (
            root / "agent" / "application" / "capability_acceptance_planner.py"
        )
        domain_path = root / "agent" / "domain" / "action_capabilities.py"
        self.assertFalse((root / "capability_acceptance_planner.py").exists())
        self.assertTrue(application_path.is_file())
        self.assertIs(PROMOTABLE_ACTIONS, planner.PROMOTABLE_ACTIONS)
        self.assertIs(PROMOTABLE_ACTIONS, capability_acceptance.PROMOTABLE_ACTIONS)
        self.assertIs(
            PROMOTABLE_ACTIONS,
            capability_acceptance_runtime.PROMOTABLE_ACTIONS,
        )
        self.assertIs(
            CALIBRATION_BOUND_ACTIONS,
            capability_acceptance.CALIBRATION_BOUND_ACTIONS,
        )
        self.assertIs(
            CALIBRATION_BOUND_ACTIONS,
            capability_acceptance_runtime.CALIBRATION_BOUND_ACTIONS,
        )
        self.assertEqual(
            {
                "tap_semantic",
                "dismiss_overlay",
                "swipe",
                "back",
                "home",
                "reveal_system_navigation",
                "input_verified_text",
                "double_tap",
                "long_press",
                "drag",
            },
            set(PROMOTABLE_ACTIONS),
        )
        self.assertEqual(
            {"double_tap", "long_press", "drag", "reveal_system_navigation"},
            set(CALIBRATION_BOUND_ACTIONS),
        )

        application_source = application_path.read_text(encoding="utf-8")
        domain_source = domain_path.read_text(encoding="utf-8")
        self.assertEqual(1, domain_source.count("PROMOTABLE_ACTIONS = frozenset("))
        self.assertEqual(
            1,
            domain_source.count("CALIBRATION_BOUND_ACTIONS = frozenset("),
        )
        for forbidden in (
            "agent.infrastructure",
            "capability_acceptance import",
            "fastapi",
            "pydantic",
            "web_app",
            "from pathlib",
            "Path(",
        ):
            self.assertNotIn(forbidden, application_source)

        legacy_imports: list[str] = []
        legacy_action_authorities: list[str] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module == "capability_acceptance_planner"
                ):
                    legacy_imports.append(str(path.relative_to(root)))
                if isinstance(node, ast.Import):
                    for item in node.names:
                        if item.name == "capability_acceptance_planner":
                            legacy_imports.append(str(path.relative_to(root)))
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module == "capability_acceptance"
                    and any(
                        alias.name
                        in {"PROMOTABLE_ACTIONS", "CALIBRATION_BOUND_ACTIONS"}
                        for alias in node.names
                    )
                ):
                    legacy_action_authorities.append(str(path.relative_to(root)))
        self.assertEqual([], legacy_imports)
        self.assertEqual([], legacy_action_authorities)

    def test_deepseek_task_graph_has_one_application_entry(self) -> None:
        import agent.application.deepseek_task_graph as deepseek_task_graph
        import agent.application.capability_acceptance_planner as capability_acceptance_planner
        import agent.application.universal_agent_orchestrator as universal_agent_orchestrator
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
        self.assertNotIn("LocalRiskPolicyConfig", application_source)
        self.assertNotIn("self.semantic_risk_policy", application_source)
        self.assertNotIn("semantic_risk_policy:", application_source)
        self.assertNotIn("@dataclass", application_source)
        self.assertNotIn("class DynamicTaskGraph", application_source)
        self.assertNotIn("def _graph_from_payload", application_source)

        web_source = (root / "web_app.py").read_text(encoding="utf-8")
        self.assertNotIn("load_local_risk_policy", web_source)
        self.assertNotIn("semantic_risk_policy=", web_source)

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
        import agent.application.capability_acceptance_planner as capability_acceptance_planner
        import agent.application.universal_agent_orchestrator as universal_agent_orchestrator
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

    def test_vision_model_contract_has_one_layered_identity(self) -> None:
        import agent.infrastructure.generic_scene_observer as generic_scene_observer
        import agent.infrastructure.dashscope_vision_provider as vision_agent
        import agent.infrastructure.qwen_runtime_errors as qwen_runtime_errors
        import agent.application.qwen_visual_decision as qwen_visual_decision
        from agent.domain.vision_model import (
            VisionAgentError,
            VisionModelConfig,
            public_model_identity,
        )
        from agent.infrastructure.environment_vision_model_config import (
            load_vision_model_config,
        )

        root = Path(__file__).resolve().parent
        domain_path = root / "agent" / "domain" / "vision_model.py"
        loader_path = (
            root
            / "agent"
            / "infrastructure"
            / "environment_vision_model_config.py"
        )
        provider_path = (
            root
            / "agent"
            / "infrastructure"
            / "dashscope_vision_provider.py"
        )
        runtime_errors_path = (
            root / "agent" / "infrastructure" / "qwen_runtime_errors.py"
        )
        self.assertFalse((root / "vision_model_config.py").exists())
        self.assertFalse((root / "vision_agent.py").exists())
        self.assertFalse((root / "qwen_runtime_errors.py").exists())
        self.assertTrue(domain_path.is_file())
        self.assertTrue(loader_path.is_file())
        self.assertTrue(provider_path.is_file())
        self.assertTrue(runtime_errors_path.is_file())
        self.assertIs(VisionAgentError, qwen_visual_decision.VisionAgentError)
        self.assertIs(VisionAgentError, generic_scene_observer.VisionAgentError)
        self.assertIs(VisionModelConfig, vision_agent.VisionModelConfig)
        self.assertIs(
            public_model_identity,
            qwen_visual_decision.public_model_identity,
        )
        self.assertTrue(callable(load_vision_model_config))

        domain_source = domain_path.read_text(encoding="utf-8")
        loader_source = loader_path.read_text(encoding="utf-8")
        provider_source = provider_path.read_text(encoding="utf-8")
        runtime_errors_source = runtime_errors_path.read_text(encoding="utf-8")
        for forbidden in (
            "agent.application",
            "agent.infrastructure",
            "import os",
            "os.environ",
            "httpx",
            "from PIL",
            "vision_agent",
        ):
            self.assertNotIn(forbidden, domain_source)
        self.assertEqual(1, loader_source.count("def load_vision_model_config("))
        self.assertEqual(1, loader_source.count("os.environ"))
        self.assertNotIn("class VisionAgentError", provider_source)
        self.assertEqual(1, provider_source.count("class DashScopeVisionProvider:"))
        self.assertEqual(1, runtime_errors_source.count("def classify_qwen_error("))

        legacy_imports: list[str] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module == "vision_model_config"
                ):
                    legacy_imports.append(str(path.relative_to(root)))
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module in {"vision_agent", "qwen_runtime_errors"}
                ):
                    legacy_imports.append(str(path.relative_to(root)))
                if isinstance(node, ast.Import):
                    for item in node.names:
                        if item.name in {
                            "vision_model_config",
                            "vision_agent",
                            "qwen_runtime_errors",
                        }:
                            legacy_imports.append(str(path.relative_to(root)))
        self.assertEqual([], legacy_imports)

    def test_visual_evidence_and_image_measurement_are_layered_once(self) -> None:
        import agent.infrastructure.observation_images as observation_images
        import agent.infrastructure.generic_scene_observer as generic_scene_observer
        import agent.application.qwen_visual_decision as qwen_visual_decision
        from agent.domain.visual_evidence import (
            LocalFrameStability,
            VisualObstruction,
        )

        root = Path(__file__).resolve().parent
        domain_path = root / "agent" / "domain" / "visual_evidence.py"
        infrastructure_path = (
            root / "agent" / "infrastructure" / "observation_images.py"
        )
        self.assertFalse((root / "observation_images.py").exists())
        self.assertTrue(domain_path.is_file())
        self.assertTrue(infrastructure_path.is_file())
        self.assertFalse(hasattr(qwen_visual_decision, "LocalFrameStability"))
        self.assertIs(VisualObstruction, generic_scene_observer.VisualObstruction)
        self.assertIs(
            LocalFrameStability,
            observation_images.LocalFrameStability,
        )
        self.assertIs(VisualObstruction, observation_images.VisualObstruction)
        self.assertEqual("agent.domain.visual_evidence", LocalFrameStability.__module__)
        self.assertEqual("agent.domain.visual_evidence", VisualObstruction.__module__)

        domain_source = domain_path.read_text(encoding="utf-8")
        infrastructure_source = infrastructure_path.read_text(encoding="utf-8")
        self.assertEqual(1, domain_source.count("class LocalFrameStability:"))
        self.assertEqual(1, domain_source.count("class VisualObstruction("))
        self.assertNotIn("class LocalFrameStability:", infrastructure_source)
        self.assertNotIn("class VisualObstruction:", infrastructure_source)
        for forbidden in (
            "agent.application",
            "agent.infrastructure",
            "from PIL",
            "import os",
            "Image.",
            "web_app",
            "vision_agent",
            "robot_core",
        ):
            self.assertNotIn(forbidden, domain_source)

        legacy_imports: list[str] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module == "observation_images"
                ):
                    legacy_imports.append(str(path.relative_to(root)))
                if isinstance(node, ast.Import):
                    for item in node.names:
                        if item.name == "observation_images":
                            legacy_imports.append(str(path.relative_to(root)))
        self.assertEqual([], legacy_imports)

    def test_trusted_observation_has_one_layered_identity(self) -> None:
        import agent.infrastructure.observation_images as observation_images
        import agent.infrastructure.trusted_observation_frames as frame_adapter
        import agent.infrastructure.generic_scene_observer as generic_scene_observer
        import agent.application.qwen_visual_decision as qwen_visual_decision
        import agent.application.universal_agent_orchestrator as universal_agent_orchestrator
        from agent.domain.trusted_observation import TrustedObservation

        root = Path(__file__).resolve().parent
        domain_path = root / "agent" / "domain" / "trusted_observation.py"
        adapter_path = (
            root
            / "agent"
            / "infrastructure"
            / "trusted_observation_frames.py"
        )
        qwen_path = root / "agent" / "application" / "qwen_visual_decision.py"
        observer_path = (
            root / "agent" / "infrastructure" / "generic_scene_observer.py"
        )
        self.assertTrue(domain_path.is_file())
        self.assertTrue(adapter_path.is_file())
        self.assertFalse((root / "trusted_observation.py").exists())
        self.assertFalse(hasattr(qwen_visual_decision, "TrustedObservation"))
        self.assertFalse(hasattr(universal_agent_orchestrator, "TrustedObservation"))
        self.assertEqual(
            "agent.domain.trusted_observation",
            TrustedObservation.__module__,
        )
        self.assertTrue(callable(frame_adapter.build_trusted_observation))
        self.assertTrue(
            callable(frame_adapter.validate_trusted_observation_against_frames)
        )
        self.assertTrue(callable(observation_images.local_frame_fingerprint))
        self.assertFalse(hasattr(generic_scene_observer, "_local_frame_fingerprint"))

        domain_source = domain_path.read_text(encoding="utf-8")
        adapter_source = adapter_path.read_text(encoding="utf-8")
        qwen_source = qwen_path.read_text(encoding="utf-8")
        observer_source = observer_path.read_text(encoding="utf-8")
        self.assertEqual(1, domain_source.count("class TrustedObservation:"))
        self.assertNotIn("class TrustedObservation:", qwen_source)
        self.assertNotIn("def _canonicalize_trusted_scene(", qwen_source)
        self.assertNotIn("def _trusted_target_local_candidate(", qwen_source)
        self.assertNotIn("def _local_frame_fingerprint(", observer_source)
        self.assertEqual(1, adapter_source.count("def build_trusted_observation("))
        self.assertEqual(
            1,
            adapter_source.count(
                "def validate_trusted_observation_against_frames("
            ),
        )
        for forbidden in (
            "agent.application",
            "agent.infrastructure",
            "from PIL",
            "import os",
            "import uuid",
            "Image.",
            "web_app",
            "vision_agent",
            "generic_scene_observer",
            "robot_core",
        ):
            self.assertNotIn(forbidden, domain_source)

        legacy_imports: list[str] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module == "qwen_visual_decision"
                    and any(item.name == "TrustedObservation" for item in node.names)
                ):
                    legacy_imports.append(str(path.relative_to(root)))
        self.assertEqual([], legacy_imports)

    def test_device_execution_has_one_modular_runtime_entry(self) -> None:
        root = Path(__file__).resolve().parent
        self.assertFalse((root / "device_executor.py").exists())
        self.assertFalse((root / "device_exclusivity.py").exists())

        orchestrator_source = (
            root / "agent" / "application" / "universal_agent_orchestrator.py"
        ).read_text(encoding="utf-8")
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

        orchestrator_source = (
            root / "agent" / "application" / "universal_agent_orchestrator.py"
        ).read_text(encoding="utf-8")
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
