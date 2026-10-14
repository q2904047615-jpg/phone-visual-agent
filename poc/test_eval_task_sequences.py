import copy
import unittest
from pathlib import Path
from eval_task_sequences import evaluate_manifest, load_manifest


MANIFEST = Path(__file__).with_name("evals") / "task_sequences" / "cases.json"


class TaskSequenceBenchmarkTests(unittest.TestCase):
    def test_three_app_replay_is_read_only_and_succeeds(self):
        report = evaluate_manifest(load_manifest(MANIFEST))

        self.assertEqual(report["case_count"], 3)
        self.assertEqual(report["distinct_app_scopes"], 3)
        self.assertEqual(report["passed_cases"], 3)
        self.assertEqual(report["physical_actions"], 0)
        self.assertEqual(report["remote_model_calls"], 0)
        self.assertEqual(report["action_match_rate"], 1.0)
        self.assertEqual(report["evaluation_protocol"], "2026-10-14-task-sequence-evaluation-v1")
        self.assertEqual(report["failure_categories"], {})

    def test_mismatch_is_reported_without_hardware(self):
        manifest = copy.deepcopy(load_manifest(MANIFEST))
        manifest["cases"][0]["recorded_actions"][1]["point"] = [999, 999]

        report = evaluate_manifest(manifest)

        self.assertEqual(report["failed_cases"], 1)
        self.assertEqual(report["physical_actions"], 0)
        self.assertIn("参数不匹配", report["cases"][0]["failure"])
        self.assertEqual(report["cases"][0]["failure_category"], "execution")
        self.assertEqual(report["failure_categories"], {"execution": 1})

    def test_terminal_mismatch_is_classified_as_verification_without_hardware(self):
        manifest = copy.deepcopy(load_manifest(MANIFEST))
        manifest["cases"][0]["recorded_terminal_status"] = "failed"

        report = evaluate_manifest(manifest)

        self.assertEqual(report["failed_cases"], 1)
        self.assertEqual(report["cases"][0]["failure_category"], "verification")
        self.assertEqual(report["failure_categories"], {"verification": 1})
        self.assertEqual(report["physical_actions"], 0)


if __name__ == "__main__":
    unittest.main()
