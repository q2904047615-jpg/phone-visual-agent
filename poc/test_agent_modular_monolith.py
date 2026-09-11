from __future__ import annotations
from typing import Any
from agent.domain import CanonicalSelectionReceipt
from agent.domain import ConfirmationAuthority
from agent.application import POST_ACTION_TRANSITION_PROTOCOL_VERSION
from pathlib import Path
from types import SimpleNamespace
from agent.application import UniversalAgentSessionState
import ast
import unittest


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
            step_id="authenticate",
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
            qwen_decision=SimpleNamespace(proposal=SimpleNamespace(
                status="action", action=SimpleNamespace(action="tap_semantic"))),
            confirmation_authority=authority,
            evidence_paths=["a.json", "a.json", "b.json"],
        )

        snapshot = session.snapshot()
        self.assertEqual(1, snapshot["step_number"])
        self.assertEqual(0, snapshot["physical_actions"])
        self.assertTrue(snapshot["confirmation_ready"])
        self.assertNotIn("controller_decision", session.__dataclass_fields__)
        self.assertNotIn("controller_decision", vars(session))
        self.assertEqual("tap_semantic", snapshot["controller_decision"]["canonical_class"])
        with self.assertRaises(AttributeError):
            session.controller_decision = CanonicalSelectionReceipt(allowed=False, reason="旧副本否决")
        session.qwen_decision.proposal.action.action = "home"
        self.assertEqual("home", session.snapshot()["controller_decision"]["canonical_class"])
        self.assertEqual(authority.scope(), snapshot["confirmation_scope"])
        self.assertEqual(["a.json", "b.json"], snapshot["evidence"])
        self.assertEqual([], snapshot["available_action_kinds"], 'No observation has issued an action scope yet')
        session.observation_action_kinds = frozenset({'home'})
        self.assertEqual(['home'], session.snapshot()['available_action_kinds'])
        self.assertEqual(
            POST_ACTION_TRANSITION_PROTOCOL_VERSION,
            snapshot["post_action_transition_protocol"],
        )
        authority.consumed = True
        consumed = session.snapshot()
        self.assertFalse(consumed["confirmation_ready"])
        self.assertIsNone(consumed["confirmation_scope"])
        session.confirmation_authority = None
        self.assertIsNone(session.snapshot()["controller_decision"])

        root = Path(__file__).resolve().parent
        orchestrator_source = (
            root / "agent" / "application" / "universal_agent_orchestrator.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("class UniversalAgentSessionState", orchestrator_source)
        self.assertNotIn(
            '"2026-08-16-universal-post-action-transition-v1"',
            orchestrator_source,
        )

    def test_app_surface_lineage_authority_and_snapshot_key_are_removed(self) -> None:
        root = Path(__file__).resolve().parent
        session_source = (
            root / "agent" / "application" / "runtime_session.py"
        ).read_text(encoding="utf-8")
        self.assertFalse((root / "agent" / "domain" / "app_surface_lineage.py").exists())
        self.assertNotIn("VerifiedAppSurfaceLineage", session_source)
        self.assertNotIn("verified_app_surface_lineage", session_source)

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
            "write_controller_decision": [],
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
            owners["write_controller_decision"],
        )
        self.assertEqual(
            ["_build_observation"],
            owners["trusted_observation_factory"],
        )

    def test_universal_orchestrator_and_usage_have_one_layered_entry(self) -> None:
        from agent.application.universal_agent_orchestrator import UniversalAgentOrchestrator
        from agent.application.vision_usage import VisionSessionUsageLedger
        root = Path(__file__).resolve().parent
        self.assertEqual("agent.application.universal_agent_orchestrator", UniversalAgentOrchestrator.__module__)
        self.assertEqual("agent.application.vision_usage", VisionSessionUsageLedger.__module__)
        for file in ("application/deepseek_task_graph.py", "domain/task_graph.py",
                "application/capability_acceptance_planner.py", "infrastructure/deepseek_intent_provider.py",
                "infrastructure/deepseek_failure_diagnostics.py"):
            self.assertFalse((root / "agent" / file).exists(), file)
        for file in ("universal_agent_orchestrator.py", "vision_usage.py"):
            self.assertFalse((root / file).exists())
            source = (root / "agent/application" / file).read_text(encoding="utf-8")
            for forbidden in ("agent.infrastructure", "fastapi", "pydantic", "web_app",
                    ".read_text(", ".write_text(", ".write_bytes(", "os.replace", ".mkdir("):
                self.assertNotIn(forbidden, source)

    def test_web_uses_one_session_repository_instead_of_legacy_storage(self) -> None:
        source = (Path(__file__).resolve().parent / "web_app.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("UniversalAgentSessionApplicationService", source)
        self.assertIn("InMemoryAgentSessionRepository", source)
        self.assertNotIn("generic_supervised_sessions", source)
        self.assertNotIn("generic_supervised_session_lock", source)
        self.assertNotIn("_require_generic_session_device", source)

    def test_retired_persistent_input_lineage_cannot_reenter_runtime(self) -> None:
        root = Path(__file__).resolve().parent
        retired_paths = (
            root / "input_value_lineage.py",
            root / "agent" / "domain" / "input_value_lineage.py",
            root / "agent" / "application" / "input_value_lineage.py",
            root / "agent" / "infrastructure" / "file_system_input_lineage_store.py",
        )
        for retired_path in retired_paths:
            self.assertFalse(retired_path.exists())

        retired_imports: list[str] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and (
                    "input_value_lineage" in node.module
                    or "file_system_input_lineage_store" in node.module
                ):
                    retired_imports.append(str(path.relative_to(root)))
                if isinstance(node, ast.Import):
                    for item in node.names:
                        if (
                            "input_value_lineage" in item.name
                            or "file_system_input_lineage_store" in item.name
                        ):
                            retired_imports.append(str(path.relative_to(root)))
        self.assertEqual([], retired_imports)

    def test_root_production_modules_are_only_interfaces_and_tools(self) -> None:
        root = Path(__file__).resolve().parent
        expected = {
            "agent_api_cli.py",
            "capture_click_burst.py",
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

    def test_companion_ime_runtime_is_permanently_removed(self) -> None:
        self.assertFalse((Path(__file__).resolve().parent.parent / 'android' / 'companion-ime').exists(),
            '退役Android源码工程不得恢复到正式项目目录')
        root = Path(__file__).resolve().parent
        retired = (
            root / "companion_ime_setup.py",
            root / "companion_ime_registry.example.json",
            root / "agent" / "infrastructure" / "companion_ime_runtime.py",
            root / "agent" / "infrastructure" / "companion_ime_transport.py",
            root / "agent" / "infrastructure" / "windows_companion_pairing_store.py",
        )
        self.assertTrue(all(not path.exists() for path in retired))
        production = "\n".join(path.read_text(encoding="utf-8") for path in
            (root / "agent").rglob("*.py")) + (root / "web_app.py").read_text(encoding="utf-8")
        for marker in ("companion_ime", "CompanionIme", "ROBOT_COMPANION"):
            self.assertNotIn(marker, production)

        web_source = (root / "web_app.py").read_text(encoding="utf-8")
        client_source = (root / "local_agent_api_client.py").read_text(
            encoding="utf-8"
        )
        cli_source = (root / "agent_api_cli.py").read_text(encoding="utf-8")
        self.assertIn("from fastapi import", web_source)
        self.assertIn('"/openapi.json"', client_source)
        self.assertIn("LocalAgentApiClient", cli_source)

        expected = {
            "agent_api_cli.py",
            "capture_click_burst.py",
            "eval_qwen_visual_decision.py",
            "eval_task_sequences.py",
            "local_agent_api_client.py",
            "run_xy_calibration.py",
            "touch_calibration_server.py",
            "web_app.py",
        }
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


if __name__ == "__main__":
    unittest.main()
