from __future__ import annotations
from pathlib import Path
import ast
import unittest


class AgentDependencyBoundaryTests(unittest.TestCase):
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
            is_current_test = path.name == Path(__file__).name
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
                    not is_current_test
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
        self.assertNotIn("def qwerty_keyboard_config_from_anchors(", source)
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
            is_current_test = path.name == Path(__file__).name
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
                    not is_current_test
                    and isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.value.startswith("robot_core.")
                ):
                    legacy_patches.append(str(path.relative_to(root)))
        self.assertEqual([], legacy_imports)
        self.assertEqual([], legacy_patches)

    def test_camera_coordination_has_one_infrastructure_implementation(self) -> None:
        root = Path(__file__).resolve().parent
        web_source = (root / "agent" / "bootstrap" / "runtime.py").read_text(encoding="utf-8")
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
        web_source = (root / "agent" / "bootstrap" / "runtime.py").read_text(encoding="utf-8")
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
            "promotable_actions=PROMOTABLE_ACTION_KINDS",
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
        web_source = (root / "agent" / "bootstrap" / "runtime.py").read_text(encoding="utf-8")
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
            is_current_test = path.name == Path(__file__).name
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
                    not is_current_test
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
                        and value.value == "qwen3-vl-plus"
                    ):
                        model_literal_authorities.append(str(path.relative_to(root)))
        self.assertEqual([], legacy_imports)
        self.assertEqual([], legacy_patches)
        self.assertEqual(
            [str(Path("agent") / "domain" / "vision_model.py")],
            model_literal_authorities,
        )

    def test_generic_scene_observer_has_one_infrastructure_entry(self) -> None:
        import agent.infrastructure.generic_scene_observer as observer_module
        from agent.infrastructure.generic_scene_observer import (
            SingleStepGenericSceneObserver,
        )

        root = Path(__file__).resolve().parent
        infrastructure_path = (
            root / "agent" / "infrastructure" / "generic_scene_observer.py"
        )
        self.assertFalse((root / "generic_scene_observer.py").exists())
        self.assertTrue(infrastructure_path.is_file())
        self.assertEqual(
            "agent.infrastructure.generic_scene_observer",
            SingleStepGenericSceneObserver.__module__,
        )
        self.assertEqual(
            "2026-09-02-single-step-scene-action-finish-v9",
            observer_module.SINGLE_STEP_SCENE_OBSERVER_VERSION,
        )

        source = infrastructure_path.read_text(encoding="utf-8")
        self.assertEqual(1, source.count("class SingleStepGenericSceneObserver("))
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
            PROMOTABLE_ACTION_KINDS,
        )

        root = Path(__file__).resolve().parent
        infrastructure_path = (
            root / "agent" / "infrastructure" / "capability_acceptance.py"
        )
        self.assertFalse((root / "capability_acceptance.py").exists())
        self.assertTrue(infrastructure_path.is_file())
        self.assertIs(PROMOTABLE_ACTION_KINDS, acceptance.PROMOTABLE_ACTION_KINDS)
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
            is_current_test = path.name == Path(__file__).name
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
                    not is_current_test
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
            is_current_test = path.name == Path(__file__).name
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
                    not is_current_test
                    and isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.value.startswith("capability_acceptance_runtime.")
                ):
                    legacy_patches.append(str(path.relative_to(root)))
        self.assertEqual([], legacy_imports)
        self.assertEqual([], legacy_patches)

    def test_visual_evidence_and_image_measurement_are_layered_once(self) -> None:
        import agent.infrastructure.observation_images as observation_images
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

        web_source = (root / "agent" / "bootstrap" / "runtime.py").read_text(encoding="utf-8")
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
    unittest.main()
