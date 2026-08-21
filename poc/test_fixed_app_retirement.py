from __future__ import annotations

import importlib
import json
import subprocess
import sys
import unittest


RETIRED_RUNTIME_MODULES = {
    "operation_specs",
    "task_orchestrator",
    "state_controller",
    "douyin_page_signals",
    "supervised_semantic_runtime",
    "semantic_action_adapter",
    "live_semantic_dry_run",
    "semantic_executor",
    "generic_intent",
}

RETIRED_PATHS = {
    "/api/tasks",
    "/api/tasks/{task_id}",
    "/api/tasks/{task_id}/confirm",
    "/api/tasks/{task_id}/cancel",
    "/api/agent/parse",
    "/api/agent/generic-goal",
    "/api/agent/plan-preview",
    "/api/agent/supervised/start",
    "/api/agent/supervised/{session_id}",
    "/api/agent/supervised/{session_id}/step",
    "/api/agent/supervised/{session_id}/cancel",
    "/api/agent/dry-run-step",
    "/api/agent/execute-ensure-app-step",
    "/api/agent/execute-observe-step",
    "/api/agent/execute-tap-heart-step",
    "/api/events",
}


class FixedAppRetirementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.web_app = importlib.import_module("web_app")

    def test_formal_openapi_has_no_fixed_app_routes(self) -> None:
        paths = set(self.web_app.app.openapi()["paths"])
        self.assertTrue(RETIRED_PATHS.isdisjoint(paths))
        self.assertIn("/api/agent/generic-supervised/start", paths)

    def test_runtime_has_no_fixed_app_workers_or_stores(self) -> None:
        runtime = self.web_app.runtime
        for attribute in (
            "store",
            "worker",
            "jobs",
            "agent",
            "generic_intent_parser",
            "generic_orchestrator",
            "state_observer",
            "state_runner",
            "supervised_sessions",
        ):
            self.assertFalse(hasattr(runtime, attribute), attribute)

    def test_importing_formal_service_does_not_load_retired_modules(self) -> None:
        code = (
            "import json,sys,web_app; "
            f"names={sorted(RETIRED_RUNTIME_MODULES)!r}; "
            "print(json.dumps([name for name in names if name in sys.modules]))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual([], json.loads(completed.stdout.strip()))

    def test_device_contract_reports_single_current_authority(self) -> None:
        payload = self.web_app.device()
        architecture = payload["execution_architecture"]
        self.assertTrue(architecture["fixed_app_workflows_retired"])
        self.assertEqual("universal_agent", architecture["active_orchestrator"])
        self.assertNotIn("background_compatibility_worker", architecture)
        self.assertNotIn("generic_orchestrator", architecture)
        self.assertEqual(
            "2026-08-20-deepseek-typed-task-graph-v4",
            architecture["universal_agent"]["goal_protocol"],
        )


if __name__ == "__main__":
    unittest.main()
