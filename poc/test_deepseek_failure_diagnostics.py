from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent.infrastructure.deepseek_failure_diagnostics import (
    MAX_REDACTED_DEEPSEEK_RESPONSE_CHARS,
    persist_deepseek_failure_diagnostic,
)


class DeepSeekFailureDiagnosticTests(unittest.TestCase):
    def test_persists_bounded_redacted_raw_response(self) -> None:
        raw = json.dumps(
            {
                "authorization": "Bearer live-secret",
                "api_key": "sk-example123456789",
                "image": "data:image/png;base64,AAAABBBBCCCC",
                "completion_conditions": ["tap the visible result"],
                "padding": "x" * 17000,
            }
        )
        planner = SimpleNamespace(last_raw_response=raw)
        error = RuntimeError(
            "DeepSeek 高层任务图包含低层动作表达："
            "completion_conditions.evidence_required"
        )

        with tempfile.TemporaryDirectory() as temp:
            paths = persist_deepseek_failure_diagnostic(
                planner,
                evidence_dir=Path(temp),
                prefix="../initial graph",
                failed_stage="initial_task_graph",
                error=error,
            )
            artifact = json.loads(Path(paths[0]).read_text(encoding="utf-8"))

        self.assertEqual("low_level_instruction", artifact["error_type"])
        self.assertEqual("high_level_task_planner", artifact["model_role"])
        self.assertEqual("deepseek", artifact["provider"])
        self.assertEqual(hashlib.sha256(raw.encode()).hexdigest(), artifact["raw_response_sha256"])
        self.assertEqual(len(raw), artifact["raw_response_length"])
        self.assertTrue(artifact["redacted_response_truncated"])
        self.assertLessEqual(
            len(artifact["redacted_raw_response"]),
            MAX_REDACTED_DEEPSEEK_RESPONSE_CHARS,
        )
        serialized = json.dumps(artifact, ensure_ascii=False)
        self.assertNotIn("live-secret", serialized)
        self.assertNotIn("sk-example123456789", serialized)
        self.assertNotIn("AAAABBBBCCCC", serialized)
        self.assertIn("[REDACTED_SECRET]", serialized)
        self.assertIn("[REDACTED_IMAGE_DATA_URL]", serialized)
        self.assertEqual("initial_task_graph", artifact["failed_stage"])
        self.assertTrue(Path(paths[0]).name.endswith("_deepseek_failure.json"))

    def test_without_raw_response_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths = persist_deepseek_failure_diagnostic(
                SimpleNamespace(last_raw_response=""),
                evidence_dir=Path(temp),
                prefix="initial",
                failed_stage="initial_task_graph",
                error=RuntimeError("invalid"),
            )
            self.assertEqual((), paths)
            self.assertEqual([], list(Path(temp).iterdir()))

    def test_structured_diff_whitelists_goal_fields_and_redacts_secrets(self) -> None:
        previous = SimpleNamespace(
            to_dict=lambda: {
                "goal": {
                    "objective": "依次填写两个字段",
                    "target_apps": [
                        {"app_id": "current", "app_name": "当前应用"}
                    ],
                    "entities": {
                        "input_fields": [
                            {"field_id": "subject", "field_label": "主题", "text": "first"},
                            {"field_id": "body", "field_label": "正文", "text": "second"},
                        ]
                    },
                },
                "subgoals": [
                    {
                        "subgoal_id": "fill_subject",
                        "objective": "填写主题",
                        "status": "active",
                        "depends_on": [],
                        "constraints": [],
                        "completion_conditions": ["主题为 first"],
                        "effect_ids": [],
                        "execution_class": "navigate",
                    }
                ],
            }
        )
        raw = json.dumps(
            {
                "authorization": "Bearer secret-value",
                "goal": {
                    "objective": "依次填写两个字段",
                    "target_apps": [
                        {
                            "app_id": "current",
                            "app_name": "当前应用",
                            "token": "hidden-app-token",
                        }
                    ],
                    "entities": {
                        "input_fields": [
                            {"field_id": "body", "field_label": "正文", "text": "second"},
                            {"field_id": "subject", "field_label": "主题", "text": "first"},
                        ],
                        "shell": "must-not-enter-structured-diff",
                    },
                },
                "subgoals": previous.to_dict()["subgoals"],
            },
            ensure_ascii=False,
        )

        with tempfile.TemporaryDirectory() as temp:
            paths = persist_deepseek_failure_diagnostic(
                SimpleNamespace(last_raw_response=raw),
                evidence_dir=Path(temp),
                prefix="post_action_replan",
                failed_stage="post_action_replan",
                error=RuntimeError("goal changed"),
                previous_graph=previous,
            )
            artifact = json.loads(Path(paths[0]).read_text(encoding="utf-8"))

        structured = artifact["structured_candidate_diff"]
        self.assertEqual(["input_fields"], structured["changed_fields"])
        structured_text = json.dumps(structured, ensure_ascii=False)
        self.assertNotIn("authorization", structured_text)
        self.assertNotIn("shell", structured_text)
        self.assertNotIn("hidden-app-token", structured_text)
        self.assertNotIn("secret-value", json.dumps(artifact, ensure_ascii=False))

    def test_retired_semantic_shadow_is_not_serialized(self) -> None:
        shadow = SimpleNamespace(
            to_dict=lambda: {
                "authoritative": False,
                "execution_allowed": False,
                "semantic_digest": "abc123",
                "token": "must-not-leak",
            }
        )
        planner = SimpleNamespace(
            last_raw_response='{"status":"ready"}',
            last_semantic_shadow=shadow,
            last_semantic_shadow_error="",
        )

        with tempfile.TemporaryDirectory() as temp:
            paths = persist_deepseek_failure_diagnostic(
                planner,
                evidence_dir=Path(temp),
                prefix="initial",
                failed_stage="initial_task_graph",
                error=RuntimeError("formal validation failed"),
            )
            artifact = json.loads(Path(paths[0]).read_text(encoding="utf-8"))

        self.assertNotIn("semantic_shadow", artifact)
        self.assertNotIn("must-not-leak", json.dumps(artifact, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
