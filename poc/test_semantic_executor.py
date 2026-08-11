import unittest

from semantic_executor import (
    ActionResult,
    FakeTransition,
    OfflineSemanticLoop,
    PageObservation,
    ScriptedFakeEnvironment,
    SingleStepSemanticExecutor,
)
from task_orchestrator import GenericTaskOrchestrator


def obs(
    page_state: str,
    *,
    confidence: float = 1.0,
    stable: bool = True,
    **properties,
) -> PageObservation:
    return PageObservation(
        page_state=page_state,
        properties=properties,
        confidence=confidence,
        stable=stable,
    )


class SingleStepSemanticExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orchestrator = GenericTaskOrchestrator()

    def test_douyin_loop_handles_live_and_already_liked_video(self) -> None:
        plan = self.orchestrator.compile(
            "douyin.batch_interact",
            {"target_count": 3, "like": True, "comment": False},
        )
        v1_unliked = obs("douyin_video", video_id="v1", heart_state="unliked")
        v1_liked = obs("douyin_video", video_id="v1", heart_state="liked")
        live = obs("douyin_live_preview", video_id="live-1")
        v2_liked = obs("douyin_video", video_id="v2", heart_state="liked")
        v3_unliked = obs("douyin_video", video_id="v3", heart_state="unliked")
        v3_liked = obs("douyin_video", video_id="v3", heart_state="liked")
        environment = ScriptedFakeEnvironment(
            obs("android_home"),
            (
                FakeTransition("ensure_app", v1_unliked),
                FakeTransition("observe", v1_unliked),
                FakeTransition("tap_semantic", v1_liked, node_id="tap_heart"),
                FakeTransition("observe", v1_liked, node_id="verify_heart"),
                FakeTransition("record_verified_result", v1_liked),
                FakeTransition("swipe", live),
                FakeTransition("observe", live),
                FakeTransition("swipe", v2_liked),
                FakeTransition("observe", v2_liked),
                FakeTransition("record_verified_result", v2_liked),
                FakeTransition("swipe", v3_unliked),
                FakeTransition("observe", v3_unliked),
                FakeTransition("tap_semantic", v3_liked, node_id="tap_heart"),
                FakeTransition("observe", v3_liked, node_id="verify_heart"),
                FakeTransition("record_verified_result", v3_liked),
                FakeTransition("finish", v3_liked),
            ),
        )

        report = OfflineSemanticLoop().run(
            SingleStepSemanticExecutor(plan),
            environment,
        )

        self.assertEqual(report.status, "finished", report.reason)
        self.assertEqual(report.counters["processed_videos"], 3)
        self.assertEqual(report.remaining_fake_transitions, 0)
        action_names = [action.action for action in report.actions]
        self.assertEqual(action_names.count("tap_semantic"), 2)
        self.assertEqual(action_names.count("recover_unknown"), 0)
        self.assertEqual(action_names.count("swipe"), 3)

    def test_full_live_room_uses_recovery_instead_of_swipe(self) -> None:
        plan = self.orchestrator.compile(
            "douyin.batch_interact",
            {"target_count": 1, "like": True, "comment": False},
        )
        live_room = obs("douyin_live_room", video_id="live-room-1")
        environment = ScriptedFakeEnvironment(
            live_room,
            (
                FakeTransition("ensure_app", live_room),
                FakeTransition("observe", live_room),
                FakeTransition("recover_unknown", live_room),
            ),
        )

        report = OfflineSemanticLoop().run(
            SingleStepSemanticExecutor(plan),
            environment,
        )

        actions = [action.action for action in report.actions]
        self.assertEqual(actions[:3], ["ensure_app", "observe", "recover_unknown"])
        self.assertNotIn("swipe", actions)

    def test_red_heart_is_required_before_result_is_counted(self) -> None:
        plan = self.orchestrator.compile(
            "douyin.batch_interact",
            {"target_count": 1, "like": True, "comment": False},
        )
        unliked = obs("douyin_video", video_id="v1", heart_state="unliked")
        environment = ScriptedFakeEnvironment(
            obs("android_home"),
            (
                FakeTransition("ensure_app", unliked),
                FakeTransition("observe", unliked),
                FakeTransition("tap_semantic", unliked),
                FakeTransition("observe", unliked),
            ),
        )

        report = OfflineSemanticLoop().run(
            SingleStepSemanticExecutor(plan),
            environment,
        )

        self.assertEqual(report.status, "failed")
        self.assertIn("heart_state", report.reason)
        self.assertNotIn("processed_videos", report.counters)
        self.assertNotIn(
            "record_verified_result",
            [action.action for action in report.actions],
        )

    def test_low_confidence_stops_before_first_account_action(self) -> None:
        plan = self.orchestrator.compile(
            "douyin.batch_interact",
            {"target_count": 1, "like": True, "comment": False},
        )
        environment = ScriptedFakeEnvironment(
            obs("android_home", confidence=0.40),
            (),
        )

        report = OfflineSemanticLoop().run(
            SingleStepSemanticExecutor(plan),
            environment,
        )

        self.assertEqual(report.status, "failed")
        self.assertIn("置信度不足", report.reason)
        self.assertEqual(report.actions, ())

    def test_wechat_message_finishes_only_with_matching_sent_evidence(self) -> None:
        plan = self.orchestrator.compile(
            "wechat.send_text",
            {"chat_name": "文件传输助手", "text": "你好"},
        )
        home = obs("wechat_home")
        search = obs("wechat_search")
        search_typed = obs("wechat_search", active_input="文件传输助手")
        chat = obs("wechat_chat", chat_title="文件传输助手")
        message_typed = obs(
            "wechat_chat",
            chat_title="文件传输助手",
            active_input="你好",
        )
        sent = obs(
            "wechat_chat",
            chat_title="文件传输助手",
            active_input="",
            message_sent_text="你好",
        )
        environment = ScriptedFakeEnvironment(
            obs("android_home"),
            (
                FakeTransition("ensure_app", home),
                FakeTransition("observe", home),
                FakeTransition("tap_semantic", search, node_id="open_search"),
                FakeTransition("input_verified_text", search_typed),
                FakeTransition("tap_semantic", chat, node_id="open_chat"),
                FakeTransition("observe", chat, node_id="verify_chat"),
                FakeTransition("input_verified_text", message_typed),
                FakeTransition("tap_semantic", sent, node_id="send"),
                FakeTransition("observe", sent, node_id="verify_sent"),
                FakeTransition("finish", sent),
            ),
        )

        report = OfflineSemanticLoop().run(
            SingleStepSemanticExecutor(plan),
            environment,
        )

        self.assertEqual(report.status, "finished", report.reason)
        self.assertEqual(report.remaining_fake_transitions, 0)

    def test_wechat_wrong_message_evidence_stops_before_finish(self) -> None:
        plan = self.orchestrator.compile(
            "wechat.send_text",
            {"chat_name": "文件传输助手", "text": "你好"},
        )
        home = obs("wechat_home")
        search = obs("wechat_search")
        chat = obs("wechat_chat", chat_title="文件传输助手")
        wrong_sent = obs(
            "wechat_chat",
            chat_title="文件传输助手",
            active_input="",
            message_sent_text="你号",
        )
        environment = ScriptedFakeEnvironment(
            obs("android_home"),
            (
                FakeTransition("ensure_app", home),
                FakeTransition("observe", home),
                FakeTransition("tap_semantic", search),
                FakeTransition(
                    "input_verified_text",
                    obs("wechat_search", active_input="文件传输助手"),
                ),
                FakeTransition("tap_semantic", chat),
                FakeTransition("observe", chat),
                FakeTransition(
                    "input_verified_text",
                    obs(
                        "wechat_chat",
                        chat_title="文件传输助手",
                        active_input="你好",
                    ),
                ),
                FakeTransition("tap_semantic", wrong_sent),
                FakeTransition("observe", wrong_sent),
            ),
        )

        report = OfflineSemanticLoop().run(
            SingleStepSemanticExecutor(plan),
            environment,
        )

        self.assertEqual(report.status, "failed")
        self.assertIn("message_sent_text", report.reason)
        self.assertNotIn("finish", [action.action for action in report.actions])

    def test_result_for_wrong_node_is_rejected(self) -> None:
        plan = self.orchestrator.compile("douyin.search", {"keyword": "机械臂"})
        executor = SingleStepSemanticExecutor(plan)
        first = executor.start(obs("android_home"))
        self.assertEqual(first.status, "action")
        decision = executor.advance(
            ActionResult(
                node_id="not-the-pending-node",
                success=True,
                observation=obs("douyin_home"),
            )
        )
        self.assertEqual(decision.status, "failed")
        self.assertIn("不一致", decision.reason)


if __name__ == "__main__":
    unittest.main()
