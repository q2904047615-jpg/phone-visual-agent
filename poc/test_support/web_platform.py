from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from fastapi.testclient import TestClient
import web_app
from agent.infrastructure import (
    DeviceRuntimeResourceRegistry,
    DeviceTaskRegistry,
    FileSystemAgentEvidenceStore,
)
from agent.infrastructure.robot_controller import (
    MockRobotController as _MockRobotController,
    RobotController as _RobotController,
)
from agent.infrastructure.orientation_safety import (
    _mint_single_step_scene_credential,
)


class _TestDirectionCredentialMixin:
    """Keep no-hardware tests behind the same one-shot gate."""

    def _consume_physical_execution(self, action, frame):
        credential = _mint_single_step_scene_credential(
            device_id=self.device_id,
            scene_fingerprint="test-scene",
            frame=frame,
        )
        self._physical_execution_gate.arm(
            credential,
            action=action,
            scene_fingerprint="test-scene",
        )
        return super()._consume_physical_execution(action, frame)


class RobotController(_TestDirectionCredentialMixin, _RobotController):
    def __init__(self, *args, device_id="test-device", **kwargs):
        super().__init__(*args, device_id=device_id, **kwargs)


class MockRobotController(_TestDirectionCredentialMixin, _MockRobotController):
    def __init__(self, *args, device_id="test-device", **kwargs):
        super().__init__(*args, device_id=device_id, **kwargs)


web_app.RobotController = RobotController


web_app.MockRobotController = MockRobotController


class _BasePhysicalNavigationSafetyTests(unittest.TestCase):
    @staticmethod
    def _click_barrier_receipt(click_count=1):
        return {
            "version": "2026-08-19-seller-gui-click-barrier-v1",
            "channel": "left_button_atomic_click",
            "input_events_dispatched": True,

            "requested_mouse_hold_seconds": 0.35,
            "barrier_offset_pixels": 3,
            "changed_pixels": 240,
            "return_changed_pixels": 235,
            "barrier_elapsed_ms": 35.0,
            "mechanical_contact_ack": False,
            "click_count": click_count,
        }


TEST_NUMERIC_GRID_LAYOUT = {
    "type": "numeric_grid",
    "anchors": {
        "1": [304, 744],
        "3": [685, 744],
        "7": [304, 839],
        "9": [685, 839],
        "backspace": [850, 698],
    },
}


class _BaseApiEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.no_browser_patcher = patch.dict(
            "os.environ",
            {"ROBOT_WEB_NO_BROWSER": "1"},
        )
        cls.no_browser_patcher.start()
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.original_web_output_dir = web_app.WEB_OUTPUT_DIR
        web_app.WEB_OUTPUT_DIR = Path(cls.temp_dir.name) / "web_output"
        web_app.WEB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        web_app.runtime.controller = MockRobotController()
        cls.client_context = TestClient(web_app.app)
        cls.client = cls.client_context.__enter__()
        cls.headers = {"X-Control-Token": web_app.CONTROL_TOKEN}
    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_context.__exit__(None, None, None)
        web_app.WEB_OUTPUT_DIR = cls.original_web_output_dir
        cls.temp_dir.cleanup()
        cls.no_browser_patcher.stop()
    def setUp(self) -> None:
        self.device_registry_patcher = patch.object(
            web_app.runtime,
            "device_task_registry",
            DeviceTaskRegistry(),
        )
        self.device_registry_patcher.start()
        web_app.runtime.agent_session_repository.clear()
        web_app.runtime.device_runtime_resources = DeviceRuntimeResourceRegistry(
            (web_app.runtime.device_controllers.default_device_id,)
        )
    def tearDown(self) -> None:
        self.device_registry_patcher.stop()
    def _universal_api_orchestrator(self, *, device_id="phone-01"):
        from test_universal_agent_orchestrator import CountingQwen, FixtureAdapter
        from test_single_visual_loop import Observation
        from agent.application.universal_agent_orchestrator import UniversalAgentOrchestrator
        qwen = CountingQwen()
        adapter = FixtureAdapter(device_id=device_id)
        orchestrator = UniversalAgentOrchestrator(qwen_observer=qwen,
            adapter_factory=lambda _: adapter, trusted_observation_factory=Observation,
            evidence_store_factory=FileSystemAgentEvidenceStore, device_registry=DeviceTaskRegistry())
        return orchestrator, None, qwen, adapter
    def _fake_capability_manager(self, *, device_id="capability-api-device"):
        calls = []

        class Session:
            def __init__(self):
                self.physical_actions = 0
                self.status = "awaiting_confirmation"

            def snapshot(self):
                return {
                    "session_id": "capability-session-001",
                    "status": self.status,
                    "physical_actions": self.physical_actions,
                    "confirmation_scope": {
                        "session_id": "capability-session-001",
                        "task_id": "task-001",
                        "device_id": device_id,
                        "revision": 1,
                        "step_id": "subgoal-001",
                        "effect_ids": [],
                        "observation_id": "obs-001",
                        "fingerprint": "frame-001",
                        "decision_node_id": "action-node-001",
                        "action_digest": "a" * 64,
                    },
                }

        session = Session()
        trial = SimpleNamespace(
            trial_id="trial-api-001",
            device_id=device_id,
            candidate_action="drag",
            session=session,
        )
        trial.snapshot = lambda: {
            "trial_id": trial.trial_id,
            "device_id": trial.device_id,
            "candidate_action": trial.candidate_action,
            "session": session.snapshot(),
            "report": None,
            "promotion_scope": None,
            "promotion": None,
            "requires_restart": False,
        }

        class Manager:
            def __init__(self):
                self.promoted = False

            def start(self, *, device_id, candidate_action, text):
                calls.append(("start", device_id, candidate_action, text))
                return trial

            def get(self, trial_id):
                calls.append(("get", trial_id))
                if trial_id != trial.trial_id:
                    raise web_app.CapabilityAcceptanceError("不存在")
                return trial

            def confirm(self, trial_id, confirmation):
                calls.append(("confirm", trial_id, dict(confirmation)))
                if session.physical_actions:
                    raise web_app.CapabilityAcceptanceError("动作确认已使用")
                session.physical_actions = 1
                session.status = "paused"
                return {
                    "physical_actions": 1,
                    "action_outcome": "executed",
                }

            def promotion_scope(self, trial_id):
                calls.append(("promotion_scope", trial_id))
                return SimpleNamespace(
                    to_dict=lambda: {
                        "trial_id": trial.trial_id,
                        "device_id": trial.device_id,
                        "action": trial.candidate_action,
                        "report_sha256": "a" * 64,
                        "registry_sha256": "b" * 64,
                    }
                )

            def promote(self, trial_id, confirmation):
                calls.append(("promote", trial_id, dict(confirmation)))
                if self.promoted:
                    raise web_app.CapabilityAcceptanceError("晋级确认已使用")
                self.promoted = True
                return {
                    "device_id": trial.device_id,
                    "action": trial.candidate_action,
                    "requires_restart": True,
                }

            def cancel(self, trial_id):
                calls.append(("cancel", trial_id))
                session.status = "cancelled"

        return Manager(), trial, calls
