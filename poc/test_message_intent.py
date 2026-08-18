import unittest

from message_intent import (
    CanonicalMessageIntent,
    MessageIntentError,
    subgoal_binds_recipient,
)


class CanonicalMessageIntentTests(unittest.TestCase):
    def intent(self, recipient="张三", text="今晚八点见。"):
        return CanonicalMessageIntent.from_goal(
            target_apps=[{"app_id": "chat", "app_name": "聊天应用"}],
            entities={"recipient": recipient, "input_text": text},
        )

    def test_preserves_recipient_and_message_verbatim(self):
        intent = self.intent()
        self.assertEqual("张三", intent.recipient)
        self.assertEqual("今晚八点见。", intent.message_text)
        self.assertEqual("聊天应用", intent.preview()["target_apps"][0]["app_name"])

    def test_digest_binds_message_recipient_and_revision(self):
        scope = {
            "task_id": "task-1",
            "device_id": "device-1",
            "revision": 3,
            "subgoal_id": "send",
            "risk_ids": ["risk-send"],
        }
        first = self.intent().digest(**scope)
        self.assertEqual(64, len(first))
        self.assertNotEqual(first, self.intent(recipient="李四").digest(**scope))
        self.assertNotEqual(first, self.intent(text="内容变化").digest(**scope))
        self.assertNotEqual(first, self.intent().digest(**{**scope, "revision": 4}))

    def test_rejects_missing_ambiguous_or_multiline_values(self):
        cases = [
            {"recipient": "", "input_text": "你好"},
            {"recipient": " 张三", "input_text": "你好"},
            {"recipient": "张三", "input_text": ""},
            {"recipient": "张三", "input_text": "第一行\n第二行"},
        ]
        for entities in cases:
            with self.subTest(entities=entities):
                with self.assertRaises(MessageIntentError):
                    CanonicalMessageIntent.from_goal(
                        target_apps=[{"app_id": "chat", "app_name": "聊天应用"}],
                        entities=entities,
                    )

    def test_recipient_becomes_exact_only_for_bound_subgoal(self):
        self.assertFalse(subgoal_binds_recipient("张三", "聊天应用在前台可见"))
        self.assertTrue(subgoal_binds_recipient("张三", "张三的聊天页面可见"))
        self.assertTrue(
            subgoal_binds_recipient(
                "Alice",
                {"completion_conditions": ["当前聊天标题逐字显示 Alice"]},
            )
        )


if __name__ == "__main__":
    unittest.main()
