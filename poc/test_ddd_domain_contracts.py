from __future__ import annotations
from agent.domain import CANONICAL_SELECTION_RECEIPT_VERSION
from agent.domain import CanonicalSelectionReceipt
from agent.domain import ConfirmationAuthority
from agent.domain import EffectConfirmationAuthority
from pathlib import Path
import ast
from agent.domain.validation import bounds_overlap
import unittest


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

    def test_confirmation_authorities_are_domain_scoped_and_stably_sorted(self) -> None:
        action = ConfirmationAuthority(
            session_id="session-1",
            task_id="task-1",
            device_id="phone-1",
            revision=3,
            step_id="authenticate",
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
            step_id="authenticate",
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
        self.assertNotIn("CanonicalSelectionReceipt(", orchestrator_source)
        session_source = (root / "agent" / "application" / "runtime_session.py").read_text(encoding="utf-8")
        self.assertIn("CanonicalSelectionReceipt(", session_source)
        self.assertNotIn("_selection_receipt", orchestrator_source)
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

    def test_scene_and_semantic_action_have_one_domain_identity(self) -> None:
        from agent.domain.semantic_action import SemanticAction
        from agent.domain.ui_scene import UIScene

        root = Path(__file__).resolve().parent
        self.assertFalse((root / "semantic_action.py").exists())
        self.assertFalse((root / "ui_scene.py").exists())
        self.assertEqual("agent.domain.semantic_action", SemanticAction.__module__)
        self.assertEqual("agent.domain.ui_scene", UIScene.__module__)

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
            "scroll",
            "swipe_element",
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

    def test_current_frame_action_binding_has_one_domain_identity(self) -> None:
        import agent.application.qwen_visual_decision as qwen_visual_decision
        from agent.domain.canonical_action_protocol import (
            GenericStepProposal,
            bind_same_response_action,
        )

        root = Path(__file__).resolve().parent
        domain_path = (
            root / "agent" / "domain" / "canonical_action_protocol.py"
        )
        self.assertFalse((root / "canonical_action_protocol.py").exists())
        self.assertTrue(domain_path.is_file())
        self.assertIs(GenericStepProposal, qwen_visual_decision.GenericStepProposal)
        self.assertIs(
            bind_same_response_action,
            qwen_visual_decision.bind_same_response_action,
        )

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
        import agent.infrastructure.generic_action_adapter as generic_action_adapter
        import agent.application.universal_agent_orchestrator as universal_agent_orchestrator
        from agent.application import runtime_session
        from agent.domain import generic_goal
        from agent.domain.generic_goal import GenericIntentDraft

        root = Path(__file__).resolve().parent
        domain_path = root / "agent" / "domain" / "generic_goal.py"
        self.assertFalse((root / "generic_goal.py").exists())
        self.assertTrue(domain_path.is_file())
        self.assertIs(GenericIntentDraft, generic_action_adapter.GenericIntentDraft)
        self.assertIs(GenericIntentDraft, runtime_session.GenericIntentDraft)
        self.assertIs(GenericIntentDraft, universal_agent_orchestrator.GenericIntentDraft)
        self.assertFalse(hasattr(generic_goal, "_parse_json_object"))

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
            "2026-09-06-single-visual-task-v1",
            SUPPORTED_TASK_CONTEXT_PROTOCOL,
        )
        self.assertEqual("agent.domain.qwen_task_context", QwenTaskContext.__module__)

        domain_source = domain_path.read_text(encoding="utf-8")
        provider_source = provider_path.read_text(encoding="utf-8")
        self.assertEqual(1, domain_source.count("class QwenTaskContext:"))
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
            "2026-09-06-qwen-whole-task-v19",
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
            ResolvedSemanticAction,
            UniversalActionController,
            UniversalActionError,
        ):
            self.assertEqual(
                "agent.domain.universal_action_controller",
                symbol.__module__,
            )
        self.assertEqual(
            "2026-09-06-universal-adb-text-v28",
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

        production_point_path = (
            root / "agent" / "infrastructure" / "generic_action_adapter.py"
        )
        production_point_sources = source + production_point_path.read_text(
            encoding="utf-8"
        ) + (root / "web_app.py").read_text(encoding="utf-8")
        for retired_point_authority in (
            "LocalPointGrounding",
            "stable_visual_point_grounding",
            "point_grounder",
            "local_point_grounding",
        ):
            self.assertNotIn(retired_point_authority, production_point_sources)

        controller_tree = ast.parse(source)
        point_action_nodes = [
            node
            for node in ast.walk(controller_tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_point_action"
        ]
        self.assertEqual(1, len(point_action_nodes))
        point_action_source = ast.get_source_segment(source, point_action_nodes[0]) or ""
        self.assertIn('action.params.get("tap_point")', point_action_source)
        self.assertNotIn("element.center", point_action_source)

        canonical_source = (
            root / "agent" / "domain" / "canonical_action_protocol.py"
        ).read_text(encoding="utf-8")
        single_step_prompt = (
            root
            / "agent"
            / "infrastructure"
            / "prompts"
            / "single_step_observation.txt"
        ).read_text(encoding="utf-8")
        self.assertIn("MODEL_STEP_DIRECT_POINT_ACTIONS", canonical_source)
        self.assertIn('"tap_point"', canonical_source)
        self.assertIn("decision.target不得携带几何或其他动作字段", canonical_source)
        self.assertNotIn('_validate_direct_target_identity', canonical_source)
        self.assertNotIn('local_audited_input_1', single_step_prompt)
        self.assertNotIn('evidence_refs', single_step_prompt)
        self.assertIn("tap_point", single_step_prompt)
        self.assertIn("点按动作的scene.elements留空", single_step_prompt)
        self.assertNotIn("scene.elements必须是空数组", single_step_prompt)
        self.assertIn("target只含role、meaning、可选label/evidence，不含几何或states", single_step_prompt)
        self.assertIn("不得用粗框中心代替明确点击点", single_step_prompt)

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


if __name__ == "__main__":
    unittest.main()
