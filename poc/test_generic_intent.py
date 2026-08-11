import json
import unittest

from generic_intent import GenericIntentError, GenericIntentParser


class FakeProvider:
    configured = True

    def __init__(self, payload):
        self.payload = payload
        self.last_messages = None

    def chat_json(self, messages, max_tokens=500):
        self.last_messages = messages
        return json.dumps(self.payload, ensure_ascii=False)


class GenericIntentTests(unittest.TestCase):
    def test_understands_non_wechat_non_douyin_app(self):
        provider = FakeProvider(
            {
                "understood": True,
                "app_id": "calculator",
                "app_name": "计算器",
                "objective": "在计算器中计算12加30",
                "entities": {"expression": "12+30"},
                "constraints": ["不要清除历史记录"],
                "success_criteria": {"display_text": "42"},
                "account_effects": [],
                "message": "",
            }
        )
        draft = GenericIntentParser(provider).parse("打开计算器计算12+30")
        goal = draft.to_goal_spec()
        self.assertEqual(goal.app_id, "calculator")
        self.assertEqual(goal.parameters["entities"]["expression"], "12+30")
        self.assertEqual(goal.success_criteria["display_text"], "42")

    def test_account_effect_is_preserved_for_confirmation(self):
        provider = FakeProvider(
            {
                "understood": True,
                "app_id": "xiaohongshu",
                "app_name": "小红书",
                "objective": "给当前笔记点赞",
                "entities": {"target": "当前笔记"},
                "constraints": [],
                "success_criteria": {"like_state": "liked"},
                "account_effects": ["like"],
                "message": "",
            }
        )
        draft = GenericIntentParser(provider).parse("给小红书当前笔记点赞")
        self.assertEqual(draft.account_effects, ("like",))
        self.assertTrue(draft.needs_confirmation)

    def test_model_cannot_return_action_plan(self):
        provider = FakeProvider(
            {
                "understood": True,
                "app_id": "calculator",
                "app_name": "计算器",
                "objective": "点击5",
                "entities": {},
                "constraints": [],
                "success_criteria": {"display_text": "5"},
                "account_effects": [],
                "message": "",
                "steps": [{"tap": [1, 2]}],
            }
        )
        with self.assertRaisesRegex(GenericIntentError, "不允许的字段"):
            GenericIntentParser(provider).parse("计算器点击5")

    def test_unsafe_control_data_inside_entities_is_rejected(self):
        provider = FakeProvider(
            {
                "understood": True,
                "app_id": "settings",
                "app_name": "设置",
                "objective": "打开蓝牙",
                "entities": {"coordinate": [1, 2]},
                "constraints": [],
                "success_criteria": {"bluetooth": True},
                "account_effects": [],
                "message": "",
            }
        )
        with self.assertRaisesRegex(GenericIntentError, "控制字段"):
            GenericIntentParser(provider).parse("打开蓝牙")

    def test_missing_app_returns_clarification(self):
        provider = FakeProvider(
            {
                "understood": False,
                "app_id": "",
                "app_name": "",
                "objective": "",
                "entities": {},
                "constraints": [],
                "success_criteria": {},
                "account_effects": [],
                "message": "请说明要操作哪个App。",
            }
        )
        draft = GenericIntentParser(provider).parse("帮我搜一下")
        self.assertFalse(draft.understood)
        self.assertIn("哪个App", draft.message)


if __name__ == "__main__":
    unittest.main()
