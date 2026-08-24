from __future__ import annotations

import importlib
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parent

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

RETIRED_SOURCE_FILES = {
    "operation_specs.py",
    "task_orchestrator.py",
    "state_controller.py",
    "douyin_page_signals.py",
    "supervised_semantic_runtime.py",
    "semantic_action_adapter.py",
    "live_semantic_dry_run.py",
    "semantic_executor.py",
    "generic_intent.py",
    "target_locator.py",
    "vision_replay.py",
    "replay_vision_eval.py",
    "analyze_state_graph_reliability.py",
    "build_vision_history_index.py",
    "build_vision_review_queue.py",
    "web_workflows.json",
    "sequence.example.json",
    "templates/douyin_home.png",
}

RETIRED_TEST_FILES = {
    "test_operation_specs.py",
    "test_task_orchestrator.py",
    "test_state_controller.py",
    "test_douyin_page_signals.py",
    "test_supervised_semantic_runtime.py",
    "test_semantic_action_adapter.py",
    "test_live_semantic_dry_run.py",
    "test_semantic_executor.py",
    "test_generic_intent.py",
    "test_reliability.py",
    "test_vision_replay.py",
    "test_vision_review_queue.py",
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

    def test_retired_sources_and_tests_are_physically_absent(self) -> None:
        leftovers = sorted(
            relative
            for relative in RETIRED_SOURCE_FILES | RETIRED_TEST_FILES
            if (ROOT / relative).exists()
        )
        self.assertEqual([], leftovers)

    def test_current_runtime_sources_cannot_restore_fixed_app_authority(self) -> None:
        forbidden = {
            "class TaskStore",
            "class RuleAgent",
            "class HybridAgent",
            "class VisionAgentRunner",
            "RETIRED_FIXED_APP_ROUTE_NAMES",
            "def send_wechat_text",
            "def execute_douyin",
            "def detect_douyin_heart",
            "from_legacy_observation",
            "QWEN_VL_MODEL",
            "DASHSCOPE_BASE_URL",
        }
        sources = (
            "web_app.py",
            "vision_agent.py",
            "robot_core.py",
            "robot_gui_poc.py",
            "ui_scene.py",
            "vision_model_config.py",
        )
        hits = []
        for name in sources:
            text = (ROOT / name).read_text(encoding="utf-8")
            hits.extend(f"{name}:{token}" for token in forbidden if token in text)
        self.assertEqual([], sorted(hits))

    def test_retired_tests_are_not_hidden_as_skips(self) -> None:
        text = (ROOT / "test_web_platform.py").read_text(encoding="utf-8")
        self.assertNotIn("@unittest.skip", text)
        self.assertNotIn("固定 App", text)

    def test_retired_input_segmentation_authority_is_physically_absent(self) -> None:
        text = (ROOT / "text_input_utils.py").read_text(encoding="utf-8")
        for retired in (
            "split_input_segments",
            "InputAttemptState",
            "InputRecoveryCoordinator",
            "MAX_FULL_RETYPES",
        ):
            with self.subTest(retired=retired):
                self.assertNotIn(retired, text)

    def test_retired_full_qwen_action_prompt_is_physically_absent(self) -> None:
        text = (ROOT / "qwen_visual_decision.py").read_text(encoding="utf-8")
        for retired in ("def _decision_prompt(", "def _decision_retry_prompt("):
            with self.subTest(retired=retired):
                self.assertNotIn(retired, text)

    def test_robot_has_no_second_direct_latin_character_veto(self) -> None:
        text = (ROOT / "robot_core.py").read_text(encoding="utf-8")
        self.assertNotIn("英文分段包含未认证字符", text)

    def test_seller_text_dialog_transport_is_physically_absent(self) -> None:
        text = "\n".join(
            (ROOT / name).read_text(encoding="utf-8")
            for name in (
                "robot_core.py",
                "robot_gui_poc.py",
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
            "2026-08-20-deepseek-typed-task-graph-v4",
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
