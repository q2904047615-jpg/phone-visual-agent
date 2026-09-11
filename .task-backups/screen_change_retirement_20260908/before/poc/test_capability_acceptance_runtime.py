import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import agent.infrastructure.capability_acceptance_runtime as acceptance_runtime
from agent.infrastructure import (
    DeviceControllerRegistry,
    DeviceTaskRegistry,
    ProvisionalDeviceControllerError,
)
from agent.infrastructure.orientation_safety import (
    PhysicalExecutionGate,
    _mint_single_step_scene_credential,
    frame_fingerprint,
)

from agent.infrastructure.capability_acceptance import (
    CapabilityAcceptanceError,
    CapabilityRegistryPromoter,
    validate_acceptance_report,
)
from agent.infrastructure.capability_acceptance_runtime import (
    CapabilityAcceptanceManager,
)
from agent.application.action_adapter import GenericActionAdapterError
from agent.domain.action_capabilities import PROMOTABLE_ACTIONS
class ProvisionalControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.registry_path = self.root / "device_registry.json"
        self.registry_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "default_device_id": "device-a",
                    "devices": [
                        {
                            "device_id": "device-a",
                            "enabled": True,
                            "window_title": "controller-a",
                            "calibration_path": "tap-a.json",
                            "verified_actions": [
                                "tap_semantic",
                                "dismiss_overlay",
                                "swipe",
                                "back",
                                "wait_for_change",
                            ],
                        },
                        {
                            "device_id": "device-b",
                            "enabled": True,
                            "window_title": "controller-b",
                            "calibration_path": "tap-b.json",
                            "verified_actions": [
                                "tap_semantic",
                                "swipe",
                                "back",
                                "wait_for_change",
                            ],
                        },
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self.registry = DeviceControllerRegistry(
            self.registry_path,
            promotable_actions=PROMOTABLE_ACTIONS,
            mock=False,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_provisional_controller_adds_exactly_one_action_without_mutating_registry(self) -> None:
        original = self.registry.controller("device-a")
        descriptors_before = self.registry.descriptors()

        with patch("agent.infrastructure.robot_controller.seller_gui.find_window") as find_window:
            provisional = self.registry.provisional_controller("device-a", "drag")

        self.assertIsNot(provisional, original)
        self.assertFalse(original.hardware_capabilities()["drag"])
        self.assertTrue(provisional.hardware_capabilities()["drag"])
        self.assertEqual(
            provisional.verified_actions,
            original.verified_actions | {"drag"},
        )
        self.assertEqual(provisional.title, original.title)
        self.assertEqual(provisional.calibration_path, original.calibration_path)
        self.assertEqual(self.registry.descriptors(), descriptors_before)
        self.assertIs(self.registry.controller("device-a"), original)
        find_window.assert_not_called()

    def test_provisional_controller_rejects_already_verified_unknown_or_unknown_device(self) -> None:
        cases = (
            ("device-a", "back", "已经通过真机验收"),
            ("device-a", "wait_for_change", "不能进入真机能力验收"),
            ("device-a", "shell_command", "不能进入真机能力验收"),
            ("missing-device", "drag", "未登记或未启用"),
        )

        with patch("agent.infrastructure.robot_controller.seller_gui.find_window") as find_window:
            for device_id, action, message in cases:
                with self.subTest(device_id=device_id, action=action):
                    with self.assertRaisesRegex(
                        ProvisionalDeviceControllerError,
                        message,
                    ):
                        self.registry.provisional_controller(device_id, action)

        find_window.assert_not_called()


class FakeTrialSession:
    def __init__(self, session_id, device_id, run_dir, action):
        self.session_id = session_id
        self.device_id = device_id
        self.run_dir = Path(run_dir)
        self.action = action
        self.status = "awaiting_confirmation"
        self.physical_actions = 0
        self.trusted_observation = SimpleNamespace(
            observation_id="obs-before",
            fingerprint="fingerprint-before",
        )

    def snapshot(self):
        return {
            "session_id": self.session_id,
            "device_id": self.device_id,
            "status": self.status,
            "physical_actions": self.physical_actions,
            "task_graph": {"task_id": "task-001", "revision": 1},
            "trusted_observation": {
                "observation_id": self.trusted_observation.observation_id,
                "fingerprint": self.trusted_observation.fingerprint,
            },
            "proposal": {
                "status": "action",
                "action": {"action": self.action},
            },
            "confirmation_scope": {
                "session_id": self.session_id,
                "task_id": "task-001",
                "device_id": self.device_id,
                "revision": 1,
                "step_id": "subgoal-001",
                "effect_ids": [],
                "observation_id": "obs-before",
                "fingerprint": "fingerprint-before",
            "decision_node_id": "action-node",
            "action_digest": "a" * 64,
            },
        }


class FakeTrialResult:
    def __init__(self, run_dir: Path, action: str):
        self.physical_actions = 1
        self.action_outcome = "executed"
        self.resolved_action = SimpleNamespace(kind=action)
        self.before_scene = SimpleNamespace(fingerprint="fingerprint-execution-before")
        self.after_scene = SimpleNamespace(fingerprint="fingerprint-after")
        self.observation_errors = ()
        self.verification_errors = ()
        self.controller_transition_evidence = ()
        self.after_model_decision = {"previous_action_outcome": "matched"}
        self.robot_result = (
            ((2, 4), (12, 8))
            if action == "drag"
            else (8, 9)
            if action == "long_press"
            else None
        )
        self.hardware_receipt = (
            {
                "version": "2026-08-16-seller-gui-contact-barrier-v3",
                "channel": "right_button_stationary_touch",
                "seller_event_barrier_confirmed": True,
                "round_trip_position_confirmed": True,
                "hold_started_after_barrier": True,
                "requested_hold_seconds": 0.8,
                "barrier_offset_pixels": 3,
                "changed_pixels": 240,
                "return_changed_pixels": 240,
                "barrier_elapsed_ms": 35.0,
                "post_barrier_settle_seconds": 0.45,
            }
            if action == "long_press"
            else None
        )
        self.before_frame_paths = tuple(
            self._frame(run_dir / f"confirm_before_{index}.jpg", "black")
            for index in range(1, 5)
        )
        self.after_frame_paths = tuple(
            self._frame(run_dir / f"confirm_after_{index}.jpg", "white")
            for index in range(1, 5)
        )
        self.before_frames = tuple(
            Image.open(path).convert("RGB") for path in self.before_frame_paths
        )
        self.evidence = self.before_frame_paths + self.after_frame_paths
        self.orientation_credential = replace(
            _mint_single_step_scene_credential(
                device_id="device-a",
                scene_fingerprint="fingerprint-execution-before",
                frame=self.before_frames[0],
            ),
            evidence_frame_fingerprint=frame_fingerprint(self.before_frames[0]),
        )
        PhysicalExecutionGate("device-a").arm(
            self.orientation_credential,
            action=action,
            scene_fingerprint="fingerprint-execution-before",
        )

    @staticmethod
    def _frame(path: Path, color: str) -> str:
        Image.new("RGB", (16, 16), color).save(path, format="JPEG")
        return str(path)

    def to_dict(self):
        payload = {
            "resolved_action": {"kind": self.resolved_action.kind},
            "physical_actions": self.physical_actions,
            "action_outcome": self.action_outcome,
            "observation_errors": [],
            "verification_errors": list(self.verification_errors),
            "controller_transition_evidence": list(
                self.controller_transition_evidence
            ),
            "robot_result": self.robot_result,
            "hardware_receipt": self.hardware_receipt,
            "before_frame_paths": list(self.before_frame_paths),
            "after_frame_paths": list(self.after_frame_paths),
            "orientation_credential": self.orientation_credential.to_dict(),
        }
        if self.resolved_action.kind == "drag":
            payload.update(
                {
                    "resolved_action": {
                        "kind": "drag",
                        "normalized_point": [0.15, 0.25],
                        "normalized_end_point": [0.8, 0.3],
                        "hold_seconds": 0.8,
                        "path_distance": 0.6519202405202649,
                        "target_element_id": "source",
                        "destination_element_id": "destination",
                        "before_fingerprint": "fingerprint-execution-before",
                    },
                    "before_scene": self._drag_scene(
                        source_bounds=[0.1, 0.2, 0.2, 0.3],
                        fingerprint="fingerprint-execution-before",
                    ),
                    "after_scene": self._drag_scene(
                        source_bounds=[0.55, 0.2, 0.65, 0.3],
                        fingerprint="fingerprint-after",
                    ),
                }
            )
        elif self.resolved_action.kind == "long_press":
            payload.update(
                {
                    "resolved_action": {
                        "kind": "long_press",
                        "normalized_point": [0.3, 0.4],
                        "hold_seconds": 0.8,
                        "target_element_id": "item",
                        "before_fingerprint": "fingerprint-execution-before",
                    },
                    "before_scene": self._long_press_scene(
                        fingerprint="fingerprint-execution-before",
                        overlay=False,
                    ),
                    "after_scene": self._long_press_scene(
                        fingerprint="fingerprint-after",
                        overlay=True,
                    ),
                }
            )
        elif self.resolved_action.kind == "input_verified_text":
            payload.update(
                {
                    "resolved_action": {
                        "kind": "input_verified_text",
                        "text": "agent",
                        "target_element_id": "field",
                        "before_fingerprint": "fingerprint-execution-before",
                    },
                    "before_scene": {
                        "foreground_app_id": "test-app",
                        "screen_id": "input",
                        "summary": "输入前",
                        "elements": [
                            {
                                "element_id": "field",
                                "role": "input",
                                "meaning": "search_field",
                                "label": "搜索",
                                "confidence": 0.95,
                                "states": {
                                    "focused": True,
                                    "value": "",
                                    "keyboard_layout": "qwerty",
                                    "keyboard_input_mode": "direct_latin",
                                    "goal_relevant": True,
                                },
                            }
                        ],
                        "stable": True,
                        "confidence": 0.95,
                        "fingerprint": "fingerprint-execution-before",
                    },
                    "after_scene": {
                        "foreground_app_id": "test-app",
                        "screen_id": "input",
                        "summary": "输入后",
                        "elements": [
                            {
                                "element_id": "field",
                                "role": "input",
                                "meaning": "search_field",
                                "label": "搜索",
                                "confidence": 0.95,
                                "states": {"value": "agent.com"},
                            }
                        ],
                        "stable": True,
                        "confidence": 0.95,
                        "fingerprint": "fingerprint-after",
                    },
                }
            )
        return payload

    @staticmethod
    def _drag_scene(*, source_bounds, fingerprint):
        return {
            "foreground_app_id": "test-app",
            "screen_id": "board",
            "summary": "拖动场景",
            "elements": [
                {
                    "element_id": "source",
                    "role": "button",
                    "meaning": "draggable_item",
                    "label": "项目",
                    "bounds": source_bounds,
                    "confidence": 0.95,
                    "states": {},
                },
                {
                    "element_id": "destination",
                    "role": "container",
                    "meaning": "drop_zone",
                    "label": "目标",
                    "bounds": [0.7, 0.2, 0.9, 0.4],
                    "confidence": 0.95,
                    "states": {},
                },
            ],
            "stable": True,
            "confidence": 0.95,
            "fingerprint": fingerprint,
        }

    @staticmethod
    def _long_press_scene(*, fingerprint, overlay):
        return {
            "foreground_app_id": "test-app",
            "screen_id": "list",
            "summary": "长按场景",
            "elements": [
                {
                    "element_id": "item",
                    "role": "list_item",
                    "meaning": "list_item",
                    "label": "项目",
                    "bounds": [0.2, 0.3, 0.4, 0.5],
                    "confidence": 0.95,
                    "states": {},
                }
            ],
            "overlays": (["context_menu"] if overlay else []),
            "stable": True,
            "confidence": 0.95,
            "fingerprint": fingerprint,
        }


class FakeTrialOrchestrator:
    def __init__(self, registry, proposed_action, calls, confirm_error=None):
        self.registry = registry
        self.proposed_action = proposed_action
        self.calls = calls
        self.confirm_error = confirm_error

    def start(self, *, session_id, raw_goal, device_id, run_dir):
        self.calls.append(("start", raw_goal, device_id))
        self.registry.reserve(device_id, session_id)
        return FakeTrialSession(session_id, device_id, run_dir, self.proposed_action)

    def confirm_one(self, session, confirmation):
        self.calls.append(("confirm", dict(confirmation)))
        if self.confirm_error is not None:
            session.physical_actions += int(self.confirm_error.physical_actions)
            session.status = "failed"
            self.registry.release(session.device_id, session.session_id)
            raise self.confirm_error
        result = FakeTrialResult(session.run_dir, session.action)
        session.physical_actions += 1
        session.trusted_observation = SimpleNamespace(
            observation_id="obs-after",
            fingerprint="fingerprint-after",
        )
        session.status = "awaiting_confirmation"
        return result

    def cancel(self, session):
        self.calls.append(("cancel", session.session_id))
        session.status = "cancelled"
        self.registry.release(session.device_id, session.session_id)

    def pause(self, session):
        self.calls.append(("pause", session.session_id))
        session.status = "paused"
        self.registry.release(session.device_id, session.session_id)


class CapabilityAcceptanceManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.registry_path = self.root / "device_registry.json"
        self.calibration_path = self.root / "tap-a.json"
        self.calibration_path.write_text(
            json.dumps(
                {
                    "version": 2,
                    "enabled": True,
                    "validated": True,
                    "accepted_fit": True,
                    "frame_size": [540, 960],
                    "coverage": {
                        "sufficient": True,
                        "normalized_bounds": [0.05, 0.05, 0.95, 0.95],
                        "normalized_hull": [
                            [0.05, 0.05],
                            [0.95, 0.05],
                            [0.95, 0.95],
                            [0.05, 0.95],
                        ],
                    },
                    "validation": {
                        "passed": True,
                        "coverage_passed": True,
                        "coverage": {
                            "sufficient": True,
                            "normalized_bounds": [0.05, 0.05, 0.95, 0.95],
                            "normalized_hull": [
                                [0.05, 0.05],
                                [0.95, 0.05],
                                [0.95, 0.95],
                                [0.05, 0.95],
                            ],
                        },
                    },
                    "target_to_command": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                }
            ),
            encoding="utf-8",
        )
        self.registry_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "default_device_id": "device-a",
                    "devices": [
                        {
                            "device_id": "device-a",
                            "enabled": True,
                            "calibration_path": "tap-a.json",
                            "verified_actions": ["tap_semantic", "swipe", "back"],
                        },
                        {
                            "device_id": "device-b",
                            "enabled": True,
                            "calibration_path": "tap-a.json",
                            "verified_actions": ["tap_semantic", "swipe", "back"],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.device_registry = DeviceTaskRegistry()
        self.provisional_calls = []
        self.orchestrator_calls = []
        self.proposed_action = "drag"
        self.confirm_error = None

        def provisional(device_id, action):
            self.provisional_calls.append((device_id, action))
            return SimpleNamespace(
                device_id=device_id,
                candidate_action=action,
                calibration_path=self.calibration_path,
            )

        def orchestrator_factory(_controller, candidate_action):
            self.assertEqual(_controller.candidate_action, candidate_action)
            return FakeTrialOrchestrator(
                self.device_registry,
                self.proposed_action,
                self.orchestrator_calls,
                self.confirm_error,
            )

        self.manager = CapabilityAcceptanceManager(
            provisional_controller_factory=provisional,
            orchestrator_factory=orchestrator_factory,
            device_registry=self.device_registry,
            output_dir=self.root / "output",
            registry_path=self.registry_path,
            code_revision_provider=lambda: "test-revision",
            id_factory=lambda: "trial-001",
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_active_product_session_rejects_before_directory_controller_or_models(self):
        self.device_registry.reserve("device-a", "product-session")

        with self.assertRaisesRegex(CapabilityAcceptanceError, "已有活动任务"):
            self.manager.start(
                device_id="device-a",
                candidate_action="drag",
                text="拖动一个安全控件",
            )

        self.assertEqual(self.provisional_calls, [])
        self.assertEqual(self.orchestrator_calls, [])
        self.assertFalse((self.root / "output").exists())

    def test_dirty_revision_rejects_before_directory_controller_or_models(self):
        self.manager.code_revision_provider = lambda: "test-revision+dirty"

        with self.assertRaisesRegex(CapabilityAcceptanceError, "未提交修改"):
            self.manager.start(
                device_id="device-a",
                candidate_action="drag",
                text="拖动一个安全控件",
            )

        self.assertEqual(self.provisional_calls, [])
        self.assertEqual(self.orchestrator_calls, [])
        self.assertFalse((self.root / "output").exists())

    def test_restart_recovers_trial_read_only_without_authority_or_hardware(self):
        trial = self.manager.start(
            device_id="device-a",
            candidate_action="drag",
            text="拖动一个安全控件",
        )
        registry_before = self.registry_path.read_bytes()
        recovered_provisional_calls = []
        recovered_orchestrator_calls = []
        recovered = CapabilityAcceptanceManager(
            provisional_controller_factory=lambda *args: recovered_provisional_calls.append(args),
            orchestrator_factory=lambda *args: recovered_orchestrator_calls.append(args),
            device_registry=DeviceTaskRegistry(),
            output_dir=self.root / "output",
            registry_path=self.registry_path,
            code_revision_provider=lambda: "new-revision",
            id_factory=lambda: "trial-002",
        )

        snapshot = recovered.get(trial.trial_id).snapshot()

        self.assertTrue(snapshot["read_only_recovered"])
        self.assertEqual(snapshot["trial_id"], "trial-001")
        self.assertEqual(snapshot["session"]["physical_actions"], 0)
        self.assertIsNone(snapshot["promotion_scope"])
        self.assertEqual(recovered_provisional_calls, [])
        self.assertEqual(recovered_orchestrator_calls, [])
        with self.assertRaisesRegex(CapabilityAcceptanceError, "仅可查看"):
            recovered.confirm("trial-001", {})
        with self.assertRaisesRegex(CapabilityAcceptanceError, "仅可查看"):
            recovered.promotion_scope("trial-001")
        with self.assertRaisesRegex(CapabilityAcceptanceError, "仅可查看"):
            recovered.promote("trial-001", {})
        with self.assertRaisesRegex(CapabilityAcceptanceError, "仅可查看"):
            recovered.cancel("trial-001")
        self.assertEqual(self.registry_path.read_bytes(), registry_before)

    def test_mismatched_qwen_action_cancels_without_physical_action(self):
        self.proposed_action = "long_press"

        with self.assertRaisesRegex(CapabilityAcceptanceError, "不是本次候选动作"):
            self.manager.start(
                device_id="device-a",
                candidate_action="drag",
                text="拖动一个安全控件",
            )

        self.assertIsNone(self.device_registry.active_session("device-a"))
        self.assertIn(("cancel", "capability-trial-trial-001"), self.orchestrator_calls)

    def test_matching_trial_stops_at_confirmation_with_zero_actions(self):
        trial = self.manager.start(
            device_id="device-a",
            candidate_action="drag",
            text="拖动一个安全控件",
        )

        snapshot = trial.snapshot()
        self.assertEqual(snapshot["trial_id"], "trial-001")
        self.assertEqual(snapshot["candidate_action"], "drag")
        self.assertEqual(snapshot["session"]["status"], "awaiting_confirmation")
        self.assertEqual(snapshot["session"]["physical_actions"], 0)
        self.assertEqual(
            self.device_registry.active_session("device-a"),
            "capability-trial-trial-001",
        )

    def test_gesture_trial_rejects_unvalidated_calibration_before_models(self):
        payload = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        payload["validated"] = False
        self.calibration_path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(CapabilityAcceptanceError, "标定尚未启用并完成独立验证"):
            self.manager.start(
                device_id="device-a",
                candidate_action="drag",
                text="拖动一个安全控件",
            )

        self.assertEqual(self.orchestrator_calls, [])
        self.assertFalse((self.root / "output").exists())

    def test_gesture_trial_rejects_forged_top_level_validation_before_models(self):
        payload = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        payload["validation"]["passed"] = False
        self.calibration_path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(CapabilityAcceptanceError, "独立验证记录"):
            self.manager.start(
                device_id="device-a",
                candidate_action="drag",
                text="拖动一个安全控件",
            )

        self.assertEqual(self.orchestrator_calls, [])
        self.assertFalse((self.root / "output").exists())

    def test_calibration_drift_invalidates_trial_before_physical_action(self):
        trial = self.manager.start(
            device_id="device-a",
            candidate_action="drag",
            text="拖动一个安全控件",
        )
        payload = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        payload["frame_size"] = [720, 1280]
        self.calibration_path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(CapabilityAcceptanceError, "标定发生变化"):
            self.manager.confirm(
                "trial-001",
                trial.session.snapshot()["confirmation_scope"],
            )

        self.assertEqual(trial.session.physical_actions, 0)
        self.assertNotIn("confirm", [call[0] for call in self.orchestrator_calls])
        report = json.loads(trial.report_path.read_text(encoding="utf-8"))
        self.assertEqual("failed", report["status"])
        self.assertEqual(0, report["physical_actions"])
        self.assertIsNone(trial.promotion_authority)

    def test_global_stop_reaches_provisional_controller_without_action(self):
        trial = self.manager.start(
            device_id="device-a",
            candidate_action="drag",
            text="拖动一个安全控件",
        )
        trial.controller.request_stop = Mock()

        requested = self.manager.request_stop_all()

        self.assertEqual(requested, ["trial-001"])
        trial.controller.request_stop.assert_called_once_with()
        self.assertEqual(trial.session.physical_actions, 0)

    def test_confirm_writes_promotable_report_after_one_action(self):
        trial = self.manager.start(
            device_id="device-a",
            candidate_action="drag",
            text="拖动一个安全控件",
        )
        confirmation = trial.session.snapshot()["confirmation_scope"]

        result = self.manager.confirm("trial-001", confirmation)

        self.assertEqual(result.physical_actions, 1)
        report = json.loads(trial.report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["candidate_action"], "drag")
        self.assertEqual(len(report["before_frame_paths"]), 4)
        self.assertEqual(len(report["after_frame_paths"]), 4)
        self.assertIsNotNone(trial.promotion_authority)
        self.assertEqual(
            trial.promotion_authority.scope.action,
            "drag",
        )
        self.assertIsNone(self.device_registry.active_session("device-a"))
        self.assertIn(("cancel", "capability-trial-trial-001"), self.orchestrator_calls)

        with self.assertRaisesRegex(CapabilityAcceptanceError, "禁止重复执行"):
            self.manager.confirm("trial-001", confirmation)
        self.assertEqual(
            [call[0] for call in self.orchestrator_calls].count("confirm"),
            1,
        )

    def test_confirm_snapshot_failure_invalidates_live_authority(self):
        trial = self.manager.start(
            device_id="device-a",
            candidate_action="drag",
            text="拖动一个安全控件",
        )
        confirmation = trial.session.snapshot()["confirmation_scope"]
        registry_before = self.registry_path.read_bytes()
        write_json = acceptance_runtime._atomic_write_json

        def fail_promotable_snapshot(path, payload):
            if payload.get("promotion_scope") is not None:
                raise OSError("simulated trial snapshot failure")
            return write_json(path, payload)

        with patch.object(
            acceptance_runtime,
            "_atomic_write_json",
            side_effect=fail_promotable_snapshot,
        ):
            with self.assertRaisesRegex(OSError, "snapshot failure"):
                self.manager.confirm("trial-001", confirmation)

        self.assertIsNone(trial.promotion_authority)
        self.assertIsNone(trial.snapshot()["promotion_scope"])
        with self.assertRaisesRegex(CapabilityAcceptanceError, "没有可用"):
            self.manager.promotion_scope("trial-001")
        with self.assertRaisesRegex(CapabilityAcceptanceError, "没有可用"):
            self.manager.promote("trial-001", {})
        self.assertEqual(self.registry_path.read_bytes(), registry_before)

    def test_long_press_confirm_records_bounded_duration_and_visual_reobservation(self):
        self.proposed_action = "long_press"
        trial = self.manager.start(
            device_id="device-a",
            candidate_action="long_press",
            text="长按一个安全项目并观察上下文菜单。",
        )

        result = self.manager.confirm(
            "trial-001",
            trial.session.snapshot()["confirmation_scope"],
        )

        self.assertEqual(1, result.physical_actions)
        report = json.loads(trial.report_path.read_text(encoding="utf-8"))
        self.assertEqual("passed", report["status"])
        self.assertEqual(0.8, report["execution"]["resolved_action"]["hold_seconds"])
        self.assertEqual(
            ["context_menu"],
            report["execution"]["after_scene"]["overlays"],
        )
        self.assertIsNotNone(trial.promotion_authority)

    def test_report_separates_confirmed_and_execution_fresh_fingerprints(self):
        trial = self.manager.start(
            device_id="device-a",
            candidate_action="drag",
            text="拖动一个安全控件",
        )
        confirmed_scope = trial.session.snapshot()["confirmation_scope"]

        self.manager.confirm("trial-001", confirmed_scope)

        report = json.loads(trial.report_path.read_text(encoding="utf-8"))
        self.assertEqual(
            confirmed_scope["fingerprint"],
            report["before_observation"]["fingerprint"],
        )
        self.assertEqual(
            "fingerprint-execution-before",
            report["execution"]["before_scene"]["fingerprint"],
        )
        self.assertEqual(
            "fingerprint-execution-before",
            report["execution"]["resolved_action"]["before_fingerprint"],
        )

    def test_controller_drag_mismatch_is_preserved_without_reinterpretation(self):
        trial = self.manager.start(
            device_id="device-a",
            candidate_action="drag",
            text="拖动一个安全控件",
        )
        original_confirm = trial.orchestrator.confirm_one

        def confirm_with_controller_mismatch(session, confirmation):
            result = original_confirm(session, confirmation)
            result.action_outcome = "mismatched"
            result.verification_errors = ("缺少源元素向终点显著移动",)
            return result

        trial.orchestrator.confirm_one = confirm_with_controller_mismatch

        with self.assertRaisesRegex(CapabilityAcceptanceError, "未满足验收通过标准"):
            self.manager.confirm(
                "trial-001",
                trial.session.snapshot()["confirmation_scope"],
            )

        report = json.loads(trial.report_path.read_text(encoding="utf-8"))
        self.assertEqual("failed", report["status"])
        self.assertEqual("mismatched", report["action_outcome"])
        self.assertRegex(
            report["execution"]["verification_errors"][0],
            "缺少源元素向终点显著移动",
        )
        self.assertIsNone(trial.promotion_authority)

    def test_controller_mismatch_is_preserved_without_reinterpretation(self):
        self.proposed_action = "long_press"
        trial = self.manager.start(
            device_id="device-a",
            candidate_action="long_press",
            text="长按当前本地验收对象。",
        )
        confirmation = trial.session.snapshot()["confirmation_scope"]
        original_confirm = trial.orchestrator.confirm_one

        def confirm_with_controller_mismatch(session, scope):
            result = original_confirm(session, scope)
            result.action_outcome = "mismatched"
            result.verification_errors = ("长按结果不匹配",)
            return result

        trial.orchestrator.confirm_one = confirm_with_controller_mismatch

        with self.assertRaisesRegex(
            CapabilityAcceptanceError,
            "未满足验收通过标准",
        ):
            self.manager.confirm("trial-001", confirmation)

        self.assertEqual(1, trial.session.physical_actions)
        report = json.loads(trial.report_path.read_text(encoding="utf-8"))
        self.assertEqual("failed", report["status"])
        self.assertEqual("mismatched", report["action_outcome"])
        self.assertRegex(report["execution"]["verification_errors"][0], "长按结果不匹配")
        self.assertIsNone(trial.promotion_authority)

    def test_post_action_failure_records_one_action_without_promotion_or_retry(self):
        trial = self.manager.start(
            device_id="device-a",
            candidate_action="drag",
            text="拖动一个安全控件",
        )
        evidence = []
        for index in range(1, 7):
            path = trial.run_dir / f"failure_evidence_{index}.jpg"
            Image.new("RGB", (16, 16), "gray").save(path, format="JPEG")
            evidence.append(str(path))
        self.confirm_error = GenericActionAdapterError(
            "动作后观察失败",
            physical_actions=1,
            evidence=tuple(evidence),
            observation_errors=("camera timeout",),
        )
        # The trial already owns an orchestrator instance, so update that fake
        # directly to simulate the post-action error.
        trial.orchestrator.confirm_error = self.confirm_error
        confirmation = trial.session.snapshot()["confirmation_scope"]

        with self.assertRaisesRegex(GenericActionAdapterError, "动作后观察失败"):
            self.manager.confirm("trial-001", confirmation)

        report = json.loads(trial.report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["physical_actions"], 1)
        self.assertEqual(report["evidence"], evidence)
        self.assertEqual(report["observation_errors"], ["camera timeout"])
        self.assertIsNone(trial.promotion_authority)
        self.assertEqual(
            [call[0] for call in self.orchestrator_calls].count("confirm"),
            1,
        )

    def test_report_failure_after_action_is_terminal_and_releases_device(self):
        trial = self.manager.start(
            device_id="device-a",
            candidate_action="drag",
            text="拖动一个安全控件",
        )
        original_confirm = trial.orchestrator.confirm_one

        def confirm_with_missing_evidence(session, confirmation):
            result = original_confirm(session, confirmation)
            Path(result.after_frame_paths[0]).unlink()
            return result

        trial.orchestrator.confirm_one = confirm_with_missing_evidence
        confirmation = trial.session.snapshot()["confirmation_scope"]

        with self.assertRaisesRegex(CapabilityAcceptanceError, "证据无法读取"):
            self.manager.confirm("trial-001", confirmation)

        report = json.loads(trial.report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["physical_actions"], 1)
        self.assertIn("证据无法读取", report["error"])
        self.assertIsNone(self.device_registry.active_session("device-a"))
        self.assertIn(("cancel", "capability-trial-trial-001"), self.orchestrator_calls)
        self.assertIsNone(trial.promotion_authority)
        with self.assertRaisesRegex(CapabilityAcceptanceError, "禁止重复执行"):
            self.manager.confirm("trial-001", confirmation)
        self.assertEqual(
            [call[0] for call in self.orchestrator_calls].count("confirm"),
            1,
        )

    def test_report_storage_failure_after_action_still_prevents_retry(self):
        trial = self.manager.start(
            device_id="device-a",
            candidate_action="drag",
            text="拖动一个安全控件",
        )
        confirmation = trial.session.snapshot()["confirmation_scope"]

        with (
            patch.object(
                self.manager,
                "_write_pass_or_fail_report",
                side_effect=OSError("report disk full"),
            ),
            patch.object(
                self.manager,
                "_write_exception_report",
                side_effect=OSError("failure report disk full"),
            ),
        ):
            with self.assertRaisesRegex(OSError, "failure report disk full"):
                self.manager.confirm("trial-001", confirmation)

        self.assertTrue(trial.confirmation_attempted)
        self.assertFalse(trial.report_path.exists())
        self.assertEqual(trial.session.physical_actions, 1)
        self.assertIsNone(self.device_registry.active_session("device-a"))
        with self.assertRaisesRegex(CapabilityAcceptanceError, "禁止重复执行"):
            self.manager.confirm("trial-001", confirmation)
        self.assertEqual(
            [call[0] for call in self.orchestrator_calls].count("confirm"),
            1,
        )

    def test_promotion_updates_disk_once_and_records_restart_requirement(self):
        trial = self.manager.start(
            device_id="device-a",
            candidate_action="drag",
            text="拖动一个安全控件",
        )
        self.manager.confirm(
            "trial-001",
            trial.session.snapshot()["confirmation_scope"],
        )
        scope = self.manager.promotion_scope("trial-001")
        persisted_before_promotion = (
            trial.run_dir / "trial.json"
        ).read_text(encoding="utf-8")
        self.assertNotIn("promotion_authority", persisted_before_promotion)
        self.assertNotIn("source_nonce", persisted_before_promotion)
        self.assertNotIn('"promotion_receipt"', persisted_before_promotion)
        self.assertNotIn('"receipt":', persisted_before_promotion)
        self.assertNotIn("secret", persisted_before_promotion)

        result = self.manager.promote(
            "trial-001",
            scope.to_dict(),
        )

        registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        device = next(item for item in registry["devices"] if item["device_id"] == "device-a")
        self.assertIn("drag", device["verified_actions"])
        self.assertTrue(result["requires_restart"])
        snapshot = trial.snapshot()
        self.assertEqual(snapshot["promotion"], result)
        self.assertTrue(snapshot["requires_restart"])
        persisted = json.loads(
            (trial.run_dir / "trial.json").read_text(encoding="utf-8")
        )
        self.assertEqual(persisted["promotion"], result)
        self.assertTrue(persisted["requires_restart"])
        with self.assertRaisesRegex(CapabilityAcceptanceError, "已使用"):
            self.manager.promote(
                "trial-001",
                scope.to_dict(),
            )
        self.assertIsNone(trial.promotion_authority._orientation_credential)
        self.assertIsNone(trial.promotion_authority._execution_result)

    def test_cancel_releases_live_source_without_allowing_reissue(self):
        trial = self.manager.start(
            device_id="device-a",
            candidate_action="drag",
            text="拖动一个安全控件",
        )
        result = self.manager.confirm(
            "trial-001",
            trial.session.snapshot()["confirmation_scope"],
        )
        authority = trial.promotion_authority
        credential = result.orientation_credential
        registry_before = self.registry_path.read_bytes()

        self.manager.cancel("trial-001")

        self.assertTrue(authority.consumed)
        self.assertIsNone(authority._orientation_credential)
        self.assertIsNone(authority._execution_result)
        with self.assertRaisesRegex(CapabilityAcceptanceError, "live-trial"):
            CapabilityRegistryPromoter(self.registry_path).preview(
                trial.report_path,
                orientation_credential=credential,
                execution_result=result,
            )
        self.assertEqual(self.registry_path.read_bytes(), registry_before)

    def test_promotion_rejects_while_device_is_active_without_consuming_authority(self):
        trial = self.manager.start(
            device_id="device-a",
            candidate_action="drag",
            text="拖动一个安全控件",
        )
        self.manager.confirm(
            "trial-001",
            trial.session.snapshot()["confirmation_scope"],
        )
        scope = self.manager.promotion_scope("trial-001")
        self.device_registry.reserve("device-a", "product-session")

        with self.assertRaisesRegex(CapabilityAcceptanceError, "仍有活动任务"):
            self.manager.promote(
                "trial-001",
                scope.to_dict(),
            )

        self.assertFalse(trial.promotion_authority.consumed)
        registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        device = next(item for item in registry["devices"] if item["device_id"] == "device-a")
        self.assertNotIn("drag", device["verified_actions"])

    def test_code_revision_drift_consumes_promotion_authority_without_writing(self):
        trial = self.manager.start(
            device_id="device-a",
            candidate_action="drag",
            text="拖动一个安全控件",
        )
        self.manager.confirm(
            "trial-001",
            trial.session.snapshot()["confirmation_scope"],
        )
        scope = self.manager.promotion_scope("trial-001")
        self.manager.code_revision_provider = lambda: "different-revision"

        with self.assertRaisesRegex(CapabilityAcceptanceError, "代码状态发生变化"):
            self.manager.promote("trial-001", scope.to_dict())

        self.assertTrue(trial.promotion_authority.consumed)
        self.assertIsNone(trial.promotion_authority._orientation_credential)
        self.assertIsNone(trial.promotion_authority._execution_result)
        registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        device = next(item for item in registry["devices"] if item["device_id"] == "device-a")
        self.assertNotIn("drag", device["verified_actions"])


if __name__ == "__main__":
    unittest.main()
