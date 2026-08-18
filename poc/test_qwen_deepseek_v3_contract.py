from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from eval_qwen_visual_decision import _evaluate_case
from qwen_visual_decision import (
    MIGRATION_TASK_CONTEXT_PROTOCOL,
    SUPPORTED_TASK_CONTEXT_PROTOCOL,
    QwenTaskContext,
    QwenVisualDecisionObserver,
    TrustedObservation,
)
from test_qwen_visual_decision import (
    FakeProvider,
    action_payload,
    load_sequence,
    scene_for,
)
from vision_agent import VisionAgentError


ROOT = Path(__file__).resolve().parent
FIXTURE_PATH = (
    ROOT
    / "evals"
    / "qwen_visual_decision"
    / "deepseek_v3_contract_438cd22.json"
)
SOURCE_COMMIT = "438cd2258cdca681abe42da11b70c399df58063e"


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def fixture_context(name: str) -> dict:
    return copy.deepcopy(load_fixture()["contexts"][name])


def as_v2_migration(context: dict) -> dict:
    value = copy.deepcopy(context)
    value["protocol_version"] = MIGRATION_TASK_CONTEXT_PROTOCOL
    value["confirmation_gate"].pop("scope")
    return value


class FailIfCalledProvider:
    configured = True
    model = "must-not-be-called"

    def status(self) -> dict:
        return {"configured": True, "model": self.model}

    def _chat(self, *args, **kwargs) -> str:
        raise AssertionError("风险或协议门应在调用视觉模型前停止")


class DeepSeekV3QwenContractTests(unittest.TestCase):
    def test_actual_deepseek_v3_navigation_context_is_accepted(self) -> None:
        fixture = load_fixture()
        self.assertEqual(fixture["source_commit"], SOURCE_COMMIT)
        self.assertEqual(
            fixture["source_method"],
            "DynamicTaskGraph.to_qwen_context",
        )
        context = fixture["contexts"]["navigation"]
        parsed = QwenTaskContext.from_dict(context)
        self.assertEqual(
            parsed.protocol_version,
            "2026-08-11-deepseek-task-graph-v3",
        )
        self.assertEqual(parsed.task_id, context["task_id"])
        self.assertEqual(parsed.current_external_impact, "navigation_only")

    def test_v3_is_the_formal_default_and_v2_is_migration_only(self) -> None:
        self.assertEqual(
            SUPPORTED_TASK_CONTEXT_PROTOCOL,
            "2026-08-11-deepseek-task-graph-v3",
        )
        self.assertEqual(
            MIGRATION_TASK_CONTEXT_PROTOCOL,
            "2026-08-11-deepseek-task-graph-v2",
        )

    def test_v3_top_level_requires_exact_shared_fields_before_model_calls(self) -> None:
        manifest_path = ROOT / "evals" / "qwen_visual_decision" / "cases.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = {
            "protocol_version",
            "task_id",
            "device_id",
            "revision",
            "task_status",
            "goal",
            "global_constraints",
            "goal_completion_conditions",
            "current_subgoal",
            "current_external_impact",
            "risk_actions",
            "confirmation_gate",
        }
        self.assertEqual(set(fixture_context("navigation")), required)
        for mutation in ("missing", "extra"):
            with self.subTest(mutation=mutation):
                context = fixture_context("navigation")
                if mutation == "missing":
                    context.pop("task_status")
                else:
                    context["subgoals"] = []
                case = copy.deepcopy(manifest["cases"][0])
                case["id"] = f"invalid_v3_top_level_{mutation}"
                case["accepted_statuses"] = ["blocked"]
                case.pop("accepted_actions", None)
                case["task_context"] = context
                result = _evaluate_case(
                    case,
                    str(manifest_path),
                    1,
                    "contract_test_run",
                    provider_factory=FailIfCalledProvider,
                )
                self.assertEqual(result["status"], "blocked")
                self.assertEqual(
                    result["observation_diagnostics"]["model_calls"],
                    0,
                )
                self.assertEqual(
                    result["decision_diagnostics"].get("model_calls", 0),
                    0,
                )

    def test_v3_scope_requires_exact_fields(self) -> None:
        required = {"task_id", "device_id", "revision", "subgoal_id"}
        for missing in sorted(required):
            with self.subTest(missing=missing):
                context = fixture_context("navigation")
                context["confirmation_gate"]["scope"].pop(missing)
                with self.assertRaisesRegex(VisionAgentError, "confirmation_gate"):
                    QwenTaskContext.from_dict(context)

        context = fixture_context("navigation")
        context["confirmation_gate"]["scope"]["expires_at"] = "never"
        with self.assertRaisesRegex(VisionAgentError, "confirmation_gate"):
            QwenTaskContext.from_dict(context)

    def test_v3_scope_must_match_top_level_and_current_subgoal(self) -> None:
        mutations = {
            "task_id": "stale-task",
            "device_id": "stale-device",
            "revision": 999,
            "subgoal_id": "stale-subgoal",
        }
        for field, stale_value in mutations.items():
            with self.subTest(field=field):
                context = fixture_context("external_confirmed")
                context["confirmation_gate"]["scope"][field] = stale_value
                with self.assertRaisesRegex(VisionAgentError, field):
                    QwenTaskContext.from_dict(context)

    def test_v3_scope_revision_rejects_boolean_alias_for_integer_one(self) -> None:
        context = fixture_context("external_confirmed")
        self.assertEqual(context["revision"], 1)
        context["confirmation_gate"]["scope"]["revision"] = True
        with self.assertRaisesRegex(VisionAgentError, "scope.revision"):
            QwenTaskContext.from_dict(context)

    def test_v3_risk_ids_must_match_all_three_sources(self) -> None:
        for source in ("confirmation_gate", "current_subgoal", "risk_actions"):
            with self.subTest(source=source):
                context = fixture_context("external_confirmed")
                if source == "confirmation_gate":
                    context[source]["risk_ids"] = ["stale-risk"]
                elif source == "current_subgoal":
                    context[source]["risk_action_ids"] = ["stale-risk"]
                else:
                    context[source][0]["risk_id"] = "stale-risk"
                with self.assertRaisesRegex(VisionAgentError, "风险ID"):
                    QwenTaskContext.from_dict(context)

    def test_unconfirmed_v3_external_state_blocks_before_either_model(self) -> None:
        manifest_path = ROOT / "evals" / "qwen_visual_decision" / "cases.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        case = copy.deepcopy(manifest["cases"][0])
        case["id"] = "deepseek_v3_unconfirmed_contract"
        case["accepted_statuses"] = ["blocked"]
        case.pop("accepted_actions", None)
        case["task_context"] = fixture_context("external_awaiting_confirmation")
        result = _evaluate_case(
            case,
            str(manifest_path),
            1,
            "contract_test_run",
            provider_factory=FailIfCalledProvider,
        )
        self.assertTrue(result["score"]["passed"])
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["observation_diagnostics"]["model_calls"], 0)
        self.assertEqual(result["decision_diagnostics"]["model_calls"], 0)

    def test_confirmed_v3_context_can_propose_only_one_trusted_action(self) -> None:
        context = fixture_context("external_confirmed")
        frames = load_sequence("launcher_stable")
        scene = scene_for(frames)
        observation = TrustedObservation.from_scene(
            frames=frames,
            device_id=context["device_id"],
            scene=scene,
            observation_id="obs_77777777777777777777777777777777",
        )
        provider = FakeProvider(action_payload(context, observation))
        observer = QwenVisualDecisionObserver(provider)
        decision = observer.decide(
            frames=frames,
            task_context=context,
            trusted_observation=observation,
        )
        self.assertEqual(provider.calls, 1)
        self.assertEqual(decision.proposal.status, "action")
        self.assertIsNotNone(decision.proposal.action)
        self.assertNotIsInstance(decision.proposal.action, list)
        self.assertEqual(
            decision.proposal.action.params["element_id"],
            "settings_icon",
        )

    def test_v2_external_or_unknown_can_never_become_executable(self) -> None:
        external_context = as_v2_migration(
            fixture_context("external_confirmed")
        )
        parsed = QwenTaskContext.from_dict(external_context)
        self.assertFalse(parsed.external_action_allowed)
        self.assertIsNotNone(parsed.pre_observation_block_reason)

        unknown_context = as_v2_migration(
            fixture_context("external_confirmed")
        )
        unknown_context["current_external_impact"] = "unknown"
        unknown_context["current_subgoal"]["external_impact"] = "unknown"
        with self.assertRaisesRegex(VisionAgentError, "unknown"):
            QwenTaskContext.from_dict(unknown_context)

    def test_confirmed_v2_external_state_blocks_before_either_model(self) -> None:
        manifest_path = ROOT / "evals" / "qwen_visual_decision" / "cases.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        case = copy.deepcopy(manifest["cases"][0])
        case["id"] = "deepseek_v2_confirmed_external_migration"
        case["accepted_statuses"] = ["blocked"]
        case.pop("accepted_actions", None)
        case["task_context"] = as_v2_migration(
            fixture_context("external_confirmed")
        )
        result = _evaluate_case(
            case,
            str(manifest_path),
            1,
            "contract_test_run",
            provider_factory=FailIfCalledProvider,
        )
        self.assertTrue(result["score"]["passed"])
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["observation_diagnostics"]["model_calls"], 0)
        self.assertEqual(result["decision_diagnostics"]["model_calls"], 0)

    def test_v2_read_only_and_navigation_migrations_remain_usable(self) -> None:
        for impact in ("read_only", "navigation_only"):
            with self.subTest(impact=impact):
                context = as_v2_migration(fixture_context("navigation"))
                context["current_external_impact"] = impact
                context["current_subgoal"]["external_impact"] = impact
                parsed = QwenTaskContext.from_dict(context)
                self.assertEqual(parsed.protocol_version, MIGRATION_TASK_CONTEXT_PROTOCOL)
                self.assertIsNone(parsed.pre_observation_block_reason)

    def test_v3_read_only_and_navigation_contexts_remain_usable(self) -> None:
        for impact in ("read_only", "navigation_only"):
            with self.subTest(impact=impact):
                context = fixture_context("navigation")
                context["current_external_impact"] = impact
                context["current_subgoal"]["external_impact"] = impact
                parsed = QwenTaskContext.from_dict(context)
                self.assertEqual(parsed.protocol_version, SUPPORTED_TASK_CONTEXT_PROTOCOL)
                self.assertIsNone(parsed.pre_observation_block_reason)


if __name__ == "__main__":
    unittest.main()
