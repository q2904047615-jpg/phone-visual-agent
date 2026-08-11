from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analyze_state_graph_reliability import summarize_reports


class ReliabilitySummaryTests(unittest.TestCase):
    def _write_report(self, root: Path, name: str, payload: dict) -> None:
        target = root / name / "report.json"
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps(payload), encoding="utf-8")

    def test_aggregates_only_instrumented_state_graph_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_report(
                root,
                "success",
                {
                    "architecture": "page_state_graph_v1",
                    "operation": "douyin.search",
                    "success": True,
                    "reliability_metrics": {
                        "actions_issued": 3,
                        "actions_confirmed": 3,
                        "recovery_actions_issued": 1,
                        "recovery_actions_confirmed": 1,
                        "target_resolution_methods": {"windows_ocr_multiframe": 2},
                        "action_kinds": {"tap": 2, "type_text": 1},
                        "states_seen": {"douyin_search": 2},
                    },
                },
            )
            self._write_report(
                root,
                "failure",
                {
                    "architecture": "page_state_graph_v1",
                    "operation": "douyin.search",
                    "success": False,
                    "reliability_metrics": {
                        "failure_stage": "target_resolution",
                        "actions_issued": 1,
                        "actions_confirmed": 0,
                        "recovery_actions_issued": 0,
                        "recovery_actions_confirmed": 0,
                    },
                },
            )
            self._write_report(
                root,
                "legacy",
                {
                    "architecture": "page_state_graph_v1",
                    "operation": "wechat.send_text",
                    "success": True,
                },
            )

            summary = summarize_reports(root)

        self.assertEqual(summary["state_graph_reports_found"], 3)
        self.assertEqual(summary["instrumented_runs"], 2)
        self.assertEqual(summary["uninstrumented_runs"], 1)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["action_confirmation_rate"], 0.75)
        self.assertEqual(summary["recovery_confirmation_rate"], 1.0)
        self.assertEqual(summary["failure_stages"], {"target_resolution": 1})
        self.assertEqual(
            summary["target_resolution_methods"],
            {"windows_ocr_multiframe": 2},
        )
        self.assertEqual(summary["operations"]["douyin.search"]["runs"], 2)


if __name__ == "__main__":
    unittest.main()
