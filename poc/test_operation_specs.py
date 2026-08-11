from __future__ import annotations

import unittest

from operation_specs import (
    InputAttemptState,
    InputRecoveryCoordinator,
    build_operation_agent_params,
    editable_character_count,
    normalize_user_text,
    split_input_segments,
)
from vision_agent import validate_execution_plan
from web_app import RuleAgent, normalize_task_request, TaskRequest


class TextPolicyTests(unittest.TestCase):
    def test_supported_mixed_text_preserves_exact_order(self) -> None:
        text = "你好abc123，测试!"
        self.assertEqual(
            split_input_segments(text),
            ["你好", "abc123", "，", "测试", "!"],
        )

    def test_chinese_segments_are_at_most_four_characters(self) -> None:
        self.assertEqual(
            split_input_segments("今天天气真的很好"),
            ["今天天气", "真的很好"],
        )

    def test_emoji_and_newline_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "暂不支持"):
            normalize_user_text("你好🙂", field_name="正文")
        with self.assertRaisesRegex(ValueError, "不支持换行"):
            normalize_user_text("你好\n世界", field_name="正文")

    def test_editable_count_ignores_ime_pinyin_separators(self) -> None:
        self.assertEqual(editable_character_count("ni'hao"), 5)
        self.assertEqual(editable_character_count("你好1，"), 4)


class FullRetypeTests(unittest.TestCase):
    def test_wrong_segment_deletes_entire_visible_attempt(self) -> None:
        state = InputAttemptState("你好世界", ["你好", "世界"])
        self.assertTrue(state.accept_segment("你好", "你好"))
        self.assertEqual(state.begin_full_retype("你好世间"), 4)
        state.confirm_empty(True)
        self.assertEqual(state.expected_prefix, "")

    def test_empty_must_be_observed_before_restart(self) -> None:
        state = InputAttemptState("你好", ["你好"])
        state.begin_full_retype("你号")
        with self.assertRaisesRegex(ValueError, "尚未确认"):
            state.confirm_empty(False)

    def test_only_two_complete_retypes_are_allowed(self) -> None:
        state = InputAttemptState("你好", ["你好"])
        for _ in range(2):
            state.begin_full_retype("你号")
            state.confirm_empty(True)
        with self.assertRaisesRegex(ValueError, "达到2次上限"):
            state.begin_full_retype("你号")

    def test_recovery_metrics_do_not_expose_message_text(self) -> None:
        plan = [
            {"id": "step_1", "intent": "enter_text"},
            {"id": "step_2", "intent": "enter_text"},
        ]
        coordinator = InputRecoveryCoordinator(
            [
                {
                    "field_target": "消息输入框",
                    "target_text": "你好世界",
                    "segments": ["你好", "世界"],
                    "step_ids": ["step_1", "step_2"],
                    "max_full_retypes": 2,
                }
            ],
            plan,
        )
        coordinator.begin_recovery("step_1", "你号")
        metrics = coordinator.metrics()
        self.assertEqual(metrics["total_retypes"], 1)
        self.assertEqual(metrics["sessions"][0]["target_length"], 4)
        self.assertNotIn("你好世界", str(metrics))


class StructuredOperationPlanTests(unittest.TestCase):
    def _build_and_validate(self, operation: str, params: dict) -> dict:
        frozen = build_operation_agent_params(operation, params)
        validated = validate_execution_plan(
            frozen["execution_plan"],
            goal=frozen["goal"],
            allowed_texts=frozen["allowed_texts"],
        )
        self.assertEqual(validated, frozen["execution_plan"])
        InputRecoveryCoordinator(frozen["input_sessions"], validated)
        return frozen

    def test_wechat_text_plan_has_two_independent_input_sessions(self) -> None:
        frozen = self._build_and_validate(
            "wechat.send_text",
            {"chat_name": "张三", "text": "你好abc123，测试"},
        )
        self.assertEqual(len(frozen["input_sessions"]), 2)
        self.assertEqual(
            [session["target_text"] for session in frozen["input_sessions"]],
            ["张三", "你好abc123，测试"],
        )
        self.assertTrue(frozen["input_policy"]["clear_entire_attempt"])
        self.assertEqual(frozen["input_policy"]["max_full_retypes"], 2)

    def test_wechat_album_plan_accepts_first_twenty_only(self) -> None:
        frozen = self._build_and_validate(
            "wechat.send_album_image",
            {"chat_name": "文件传输助手", "image_index": 20},
        )
        media_step = next(
            step
            for step in frozen["execution_plan"]
            if step["intent"] == "select_media"
        )
        self.assertEqual(media_step["image_index"], 20)
        with self.assertRaisesRegex(ValueError, "1～20"):
            build_operation_agent_params(
                "wechat.send_album_image",
                {"chat_name": "文件传输助手", "image_index": 21},
            )

    def test_douyin_search_plan_has_verified_input_session(self) -> None:
        frozen = self._build_and_validate(
            "douyin.search", {"keyword": "机械臂测试"}
        )
        self.assertEqual(frozen["input_sessions"][0]["target_text"], "机械臂测试")
        self.assertEqual(frozen["execution_plan"][-1]["intent"], "verify_result")

    def test_douyin_batch_freezes_count_flags_and_page_cap(self) -> None:
        frozen = self._build_and_validate(
            "douyin.batch_interact",
            {
                "keyword": "机械臂",
                "target_count": 10,
                "like": True,
                "comment": True,
                "comment_text": "做得很好1",
            },
        )
        batch = frozen["execution_plan"][-1]
        self.assertEqual(batch["intent"], "interact_batch")
        self.assertEqual(batch["count"], 10)
        self.assertEqual(batch["max_pages"], 15)
        self.assertTrue(batch["like"])
        self.assertTrue(batch["comment"])
        self.assertEqual(
            [frozen["allowed_texts"][ref] for ref in batch["comment_text_refs"]],
            ["做得很好", "1"],
        )

    def test_legacy_file_transfer_is_normalized_to_specified_chat(self) -> None:
        normalized = normalize_task_request(
            TaskRequest(
                app_id="wechat",
                operation="wechat.send_text_to_file_transfer",
                params={"text": "你好"},
            )
        )
        self.assertEqual(normalized.operation, "wechat.send_text")
        self.assertEqual(
            normalized.params["source_params"]["chat_name"],
            "文件传输助手",
        )


class NewRuleAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = RuleAgent()

    def test_specified_chat_text(self) -> None:
        draft = self.agent.parse("给张三发送：你好")
        self.assertEqual(draft["operation"], "wechat.send_text")
        self.assertEqual(draft["params"], {"chat_name": "张三", "text": "你好"})

    def test_album_image(self) -> None:
        draft = self.agent.parse("给张三发送相册第3张图片")
        self.assertEqual(draft["operation"], "wechat.send_album_image")
        self.assertEqual(draft["params"]["image_index"], 3)

    def test_search_and_batch_interaction(self) -> None:
        draft = self.agent.parse("打开抖音搜索机械臂后给10个视频点赞并评论：很好")
        self.assertEqual(draft["operation"], "douyin.batch_interact")
        self.assertEqual(draft["params"]["target_count"], 10)
        self.assertTrue(draft["params"]["like"])
        self.assertTrue(draft["params"]["comment"])
        self.assertEqual(draft["params"]["comment_text"], "很好")


if __name__ == "__main__":
    unittest.main()
