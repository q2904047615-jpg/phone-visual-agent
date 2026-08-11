import json
import tempfile
import unittest
from pathlib import Path

from vision_agent import VisionDecision
from vision_replay import (
    DEFAULT_MANIFEST,
    load_case_frames,
    load_manifest,
    rescore_report,
    run_case_safely,
    score_decision,
    select_cases,
)


class VisionReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest(DEFAULT_MANIFEST)

    def test_manifest_and_images_are_valid(self) -> None:
        self.assertEqual(self.manifest["schema_version"], 1)
        self.assertEqual(len(self.manifest["cases"]), 35)
        self.assertGreaterEqual(
            sum("success" in case["tags"] for case in self.manifest["cases"]),
            7,
        )

    def test_single_historical_image_adds_four_frames_and_matching_zoom(self) -> None:
        frames = load_case_frames(self.manifest["cases"][0], DEFAULT_MANIFEST)
        self.assertEqual(len(frames), 5)
        self.assertTrue(all(frame.size == frames[0].size for frame in frames[:4]))
        self.assertEqual(frames[4].width, frames[0].width * 2)

    def test_tag_filter_selects_affected_subset(self) -> None:
        cases = select_cases(self.manifest, tags={"blur"})
        self.assertEqual(
            [case["id"] for case in cases],
            ["motion_blur_after_long_pinyin"],
        )

    def test_history_index_contains_success_and_failure_sources(self) -> None:
        index_path = DEFAULT_MANIFEST.parent / "history" / "history_index.json"
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        stats = payload["stats"]
        self.assertEqual(stats["screenshots"], 194)
        self.assertGreaterEqual(stats["successful_reports"], 8)
        self.assertGreaterEqual(stats["failed_reports"], 82)
        self.assertEqual(stats["golden"], 35)
        self.assertTrue(
            all(
                (DEFAULT_MANIFEST.parent / item["image"]).is_file()
                for item in payload["items"]
            )
        )

    def test_clear_text_recovery_matches_expected_variant(self) -> None:
        case = self.manifest["cases"][0]
        decision = VisionDecision(
            screen_type="keyboard",
            action="clear_text",
            confidence=0.99,
            reason="输入框实际为.com",
            target="评论输入框",
            observed_input_text=".com",
            delete_count=4,
            keyboard_layout={
                "type": "generic",
                "anchors": {"backspace": [870, 680]},
            },
        )
        self.assertTrue(score_decision(decision, case["expected"])["passed"])

    def test_forbidden_candidate_tap_fails_mismatch_case(self) -> None:
        case = next(
            item
            for item in self.manifest["cases"]
            if item["id"] == "pinyin_missing_first_letter_ihao"
        )
        decision = VisionDecision(
            screen_type="keyboard",
            action="tap",
            confidence=0.95,
            reason="错误地选择候选词",
            target="候选词你好",
            coordinate=(300, 580),
            observed_input_text="ihao",
        )
        score = score_decision(decision, case["expected"])
        self.assertFalse(score["passed"])
        self.assertIn("命中禁止动作", score["errors"][0])

    def test_exact_candidate_accepts_full_visible_pinyin(self) -> None:
        case = next(
            item
            for item in self.manifest["cases"]
            if item["id"] == "partial_pinyin_exact_candidate_visible"
        )
        decision = VisionDecision(
            screen_type="keyboard",
            action="tap",
            confidence=0.97,
            reason="准确候选词清晰可见",
            target="候选词三角洲行动",
            coordinate=(874, 80),
            observed_input_text="sanjiaozhouxingdong",
        )
        self.assertTrue(score_decision(decision, case["expected"])["passed"])

    def test_rescore_report_uses_existing_decisions_without_model(self) -> None:
        case = next(
            item
            for item in self.manifest["cases"]
            if item["id"] == "conflict_empty_qwerty_send_goal_type_nihao"
        )
        decision = VisionDecision(
            screen_type="keyboard",
            action="type_pinyin",
            confidence=0.98,
            reason="输入框为空，键盘清晰",
            target="微信聊天输入区域",
            text="你好",
            pinyin="nihao",
            keyboard_layout={
                "type": "qwerty",
                "anchors": {
                    "q": [100, 700],
                    "p": [900, 700],
                    "a": [140, 780],
                    "l": [860, 780],
                    "z": [210, 860],
                    "m": [790, 860],
                    "backspace": [930, 860],
                },
            },
        )
        source = {
            "schema_version": 1,
            "passed": 0,
            "total": 1,
            "total_tokens": 123,
            "results": [
                {
                    "case_id": case["id"],
                    "title": case["title"],
                    "passed": False,
                    "errors": ["旧规则失败"],
                    "cached": False,
                    "decision": decision.to_dict(),
                    "usage": {"total_tokens": 123},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.json"
            output_path = Path(directory) / "rescored.json"
            source_path.write_text(
                json.dumps(source, ensure_ascii=False), encoding="utf-8"
            )
            payload = rescore_report(
                source_path,
                manifest_path=DEFAULT_MANIFEST,
                output_path=output_path,
            )
        self.assertEqual(payload["original_passed"], 0)
        self.assertEqual(payload["passed"], 1)
        self.assertEqual(payload["new_token_consumption"], 0)
        self.assertTrue(payload["results"][0]["passed"])

    def test_invalid_model_decision_is_isolated_as_one_failed_case(self) -> None:
        class InvalidProvider:
            model = "test-model"
            last_usage = {"total_tokens": 17}

            def decide(self, **kwargs):
                raise ValueError("delete_count 与字符数不一致")

        case = self.manifest["cases"][0]
        result = run_case_safely(
            case,
            manifest_path=DEFAULT_MANIFEST,
            provider=InvalidProvider(),
            use_cache=False,
        )
        self.assertFalse(result["passed"])
        self.assertIsNone(result["decision"])
        self.assertEqual(result["usage"]["total_tokens"], 17)
        self.assertIn("delete_count", result["errors"][0])


if __name__ == "__main__":
    unittest.main()
