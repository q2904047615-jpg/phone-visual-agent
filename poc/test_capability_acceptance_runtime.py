import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from capability_acceptance import CapabilityAcceptanceError
from capability_acceptance_runtime import CapabilityAcceptanceManager
from generic_action_adapter import GenericActionAdapterError
from universal_agent_orchestrator import DeviceTaskRegistry
from web_app import DeviceControllerRegistry


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
        self.registry = DeviceControllerRegistry(self.registry_path, mock=False)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_provisional_controller_adds_exactly_one_action_without_mutating_registry(self) -> None:
        original = self.registry.controller("device-a")
        descriptors_before = self.registry.descriptors()

        with patch("robot_core.legacy.find_window") as find_window:
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

        with patch("robot_core.legacy.find_window") as find_window:
            for device_id, action, message in cases:
                with self.subTest(device_id=device_id, action=action):
                    with self.assertRaisesRegex(CapabilityAcceptanceError, message):
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
                "subgoal_id": "subgoal-001",
                "risk_ids": [],
                "observation_id": "obs-before",
                "fingerprint": "fingerprint-before",
            },
        }


class FakeTrialResult:
    def __init__(self, run_dir: Path, action: str):
        self.physical_actions = 1
        self.action_outcome = "matched"
        self.resolved_action = SimpleNamespace(kind=action)
        self.before_scene = SimpleNamespace(fingerprint="fingerprint-before-action")
        self.after_scene = SimpleNamespace(fingerprint="fingerprint-after")
        self.observation_errors = ()
        self.verification_errors = ()
        self.before_frame_paths = tuple(
            self._frame(run_dir / f"confirm_before_{index}.jpg", "black")
            for index in range(1, 5)
        )
        self.after_frame_paths = tuple(
            self._frame(run_dir / f"confirm_after_{index}.jpg", "white")
            for index in range(1, 5)
        )
        self.evidence = self.before_frame_paths + self.after_frame_paths

    @staticmethod
    def _frame(path: Path, color: str) -> str:
        Image.new("RGB", (16, 16), color).save(path, format="JPEG")
        return str(path)

    def to_dict(self):
        return {
            "resolved_action": {"kind": self.resolved_action.kind},
            "physical_actions": self.physical_actions,
            "action_outcome": self.action_outcome,
            "observation_errors": [],
            "verification_errors": [],
            "before_frame_paths": list(self.before_frame_paths),
            "after_frame_paths": list(self.after_frame_paths),
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
        self.registry_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "default_device_id": "device-a",
                    "devices": [
                        {
                            "device_id": "device-a",
                            "enabled": True,
                            "verified_actions": ["tap_semantic", "swipe", "back"],
                        },
                        {
                            "device_id": "device-b",
                            "enabled": True,
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
            return SimpleNamespace(device_id=device_id, candidate_action=action)

        def orchestrator_factory(_controller):
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
            recovered.cancel("trial-001")

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
        self.assertIn(("pause", "capability-trial-trial-001"), self.orchestrator_calls)

        with self.assertRaisesRegex(CapabilityAcceptanceError, "禁止重复执行"):
            self.manager.confirm("trial-001", confirmation)
        self.assertEqual(
            [call[0] for call in self.orchestrator_calls].count("confirm"),
            1,
        )

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
        self.assertIn(("pause", "capability-trial-trial-001"), self.orchestrator_calls)
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
        registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        device = next(item for item in registry["devices"] if item["device_id"] == "device-a")
        self.assertNotIn("drag", device["verified_actions"])


if __name__ == "__main__":
    unittest.main()
