from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from deepseek_failure_diagnostics import (
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


if __name__ == "__main__":
    unittest.main()
