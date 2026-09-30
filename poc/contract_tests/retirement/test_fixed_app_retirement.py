from __future__ import annotations
import importlib
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]

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
    "target_locator",
    "vision_replay",
    "analyze_state_graph_reliability",
    "build_vision_history_index",
    "build_vision_review_queue",
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








    def test_seller_text_dialog_transport_is_physically_absent(self) -> None:
        text = "\n".join(
            (ROOT / name).read_text(encoding="utf-8")
            for name in (
                "agent/infrastructure/robot_controller.py",
                "agent/infrastructure/seller_window_adapter.py",
                "controller_config.json",
            )
        )
        for retired in (
            "seller_text_dialog_batch",
            "direct_latin_batch",
            "vision_type_direct_latin_batch",
            "submit_direct_latin_batch",
            "_accept_owned_text_dialog",
            "GetDlgItem",
            "PostMessageW",
            "BM_CLICK",
            "IDOK",
            "TEXT_INPUT_BUTTON_X_FROM_RIGHT",
            "GW_OWNER",
        ):
            with self.subTest(retired=retired):
                self.assertNotIn(retired, text)

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
            cwd=ROOT,
        )
        self.assertEqual([], json.loads(completed.stdout.strip()))

    def test_device_contract_reports_single_current_authority(self) -> None:
        payload = self.web_app.device()
        self.assertNotIn("readiness", payload)
        self.assertTrue(all("readiness" not in item for item in payload["devices"]))
        architecture = payload["execution_architecture"]
        self.assertTrue(architecture["fixed_app_workflows_retired"])
        self.assertEqual("universal_agent", architecture["active_orchestrator"])
        self.assertEqual("universal_action_controller", architecture["controller"])
        self.assertNotIn("background_compatibility_worker", architecture)
        self.assertNotIn("generic_orchestrator", architecture)
        self.assertEqual(
            "2026-09-06-single-visual-task-v1",
            architecture["universal_agent"]["goal_protocol"],
        )

    def test_supervised_readiness_has_no_retired_queue_dependency(self) -> None:
        controller = Mock()
        controller.device_status.return_value = {
            "controller_online": True,
            "camera_online": True,
            "busy": False,
        }
        with (
            patch.object(
                self.web_app.runtime,
                "controller_for_device",
                return_value=controller,
            ),
            patch.object(
                self.web_app.runtime.vision_provider,
                "status",
                return_value={"configured": True},
            ),
        ):
            self.web_app._require_supervised_device_ready("device-local-01")


if __name__ == "__main__":
    unittest.main()
