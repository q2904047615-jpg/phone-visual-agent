from __future__ import annotations

import unittest
from types import SimpleNamespace

from PIL import Image

from operation_specs import build_state_workflow_params
from state_controller import (
    DashScopePageObserver,
    PageObservation,
    StateGraphController,
    parse_page_observation,
)
from vision_agent import VisionAgentError
from target_locator import LocalTargetResolver


def observation(state: str, **values):
    defaults = {
        "confidence": 0.95,
        "stable": True,
        "reason": "offline fixture",
    }
    if "input_text" in values or "input_is_empty" in values:
        defaults["input_scope"] = "active_input"
        defaults["input_focused"] = True
        defaults["input_bounds"] = {
            "wechat_search": (80, 40, 900, 160),
            "wechat_chat": (80, 760, 850, 930),
            "wechat_chat_keyboard": (80, 470, 850, 650),
            "douyin_search": (60, 40, 900, 170),
            "douyin_comments": (80, 650, 850, 900),
        }.get(state)
    defaults.update(values)
    return PageObservation(state=state, **defaults)


class ObservationBoundaryTests(unittest.TestCase):
    def test_observer_cannot_return_action(self):
        with self.assertRaises(VisionAgentError):
            parse_page_observation(
                {
                    "state": "android_home",
                    "confidence": 0.9,
                    "stable": True,
                    "action": "tap",
                }
            )

    def test_coordinate_object_is_normalized_and_unknown_target_is_dropped(self):
        result = parse_page_observation(
            {
                "state": "android_home",
                "confidence": 1.0,
                "stable": True,
                "targets": {
                    "douyin_icon": {"x": 782, "y": 694},
                    "comment_input": None,
                    "支付宝": {"x": 234, "y": 694},
                },
            }
        )
        self.assertEqual(result.targets["douyin_icon"], (782, 694))
        self.assertNotIn("comment_input", result.targets)
        self.assertNotIn("支付宝", result.targets)

    def test_target_box_serialization_is_normalized_to_center_and_retained(self):
        result = parse_page_observation(
            {
                "state": "android_home",
                "confidence": 1.0,
                "stable": True,
                "targets": {"douyin_icon": [768, 672, 832, 722]},
            }
        )
        self.assertEqual(result.targets["douyin_icon"], (800, 697))
        self.assertEqual(result.target_bounds["douyin_icon"], (768, 672, 832, 722))

    def test_base_page_and_keyboard_overlay_build_effective_state(self):
        result = parse_page_observation(
            {
                "base_state": "wechat_chat",
                "overlays": ["keyboard"],
                "confidence": 0.98,
                "stable": True,
                "keyboard_visible": True,
                "targets": {"chat_input": [500, 620]},
            }
        )
        self.assertEqual(result.state, "wechat_chat_keyboard")
        self.assertEqual(result.base_state, "wechat_chat")
        self.assertEqual(result.overlays, ("keyboard",))
        self.assertEqual(result.blocking_overlays, ())

    def test_comments_overlay_keeps_video_as_base(self):
        result = parse_page_observation(
            {
                "base_state": "douyin_video",
                "overlays": ["douyin_comments", "keyboard"],
                "confidence": 0.98,
                "stable": True,
                "keyboard_visible": True,
                "targets": {"comment_input": [500, 620]},
            }
        )
        self.assertEqual(result.state, "douyin_comments")
        self.assertEqual(result.base_state, "douyin_video")
        self.assertEqual(result.overlays, ("douyin_comments", "keyboard"))

    def test_known_page_blocking_overlay_without_close_fails_without_back(self):
        result = parse_page_observation(
            {
                "base_state": "wechat_chat",
                "overlays": ["permission_dialog"],
                "confidence": 0.98,
                "stable": True,
                "targets": {"chat_input": [500, 620]},
            }
        )
        action = StateGraphController().next_action(
            result,
            {
                "operation": "wechat.send_text",
                "params": {"chat_name": "文件传输助手", "text": "你好"},
            },
        )
        self.assertEqual(action.kind, "fail")
        self.assertEqual(action.transition, "")
        self.assertIn("禁止用返回键", action.reason)

    def test_known_loading_overlay_waits_then_fails_without_back(self):
        controller = StateGraphController()
        context = {
            "operation": "douyin.search",
            "params": {"keyword": "测试"},
            "search_submitted": True,
        }
        result = parse_page_observation(
            {
                "base_state": "douyin_search_results",
                "overlays": ["loading"],
                "confidence": 0.95,
                "stable": True,
            }
        )
        first = controller.next_action(result, context)
        second = controller.next_action(result, context)
        third = controller.next_action(result, context)
        self.assertEqual(first.kind, "wait")
        self.assertEqual(second.kind, "wait")
        self.assertEqual(third.kind, "fail")
        self.assertNotEqual(first.kind, "back")
        self.assertNotEqual(second.kind, "back")
        self.assertNotEqual(third.kind, "back")
        self.assertIn("加载层连续两轮未消失", third.reason)

    def test_known_loading_wait_counter_resets_after_overlay_clears(self):
        controller = StateGraphController()
        context = {
            "operation": "douyin.search",
            "params": {"keyword": "测试"},
            "search_submitted": True,
            "known_overlay_key": "douyin_search_results|loading",
            "known_overlay_waits": 2,
        }
        result = observation(
            "douyin_search_results",
            search_query_text="测试",
            search_query_verified=True,
        )
        action = controller.next_action(result, context)
        self.assertEqual(action.kind, "finish")
        self.assertNotIn("known_overlay_key", context)
        self.assertNotIn("known_overlay_waits", context)

    def test_blocking_overlay_prefers_visible_close_control(self):
        result = parse_page_observation(
            {
                "base_state": "wechat_chat",
                "overlays": ["app_dialog"],
                "confidence": 0.98,
                "stable": True,
                "targets": {"close_overlay": [920, 80]},
                "target_bounds": {"close_overlay": [890, 50, 950, 110]},
            }
        )
        action = StateGraphController().next_action(
            result,
            {
                "operation": "wechat.send_text",
                "params": {"chat_name": "文件传输助手", "text": "你好"},
            },
        )
        self.assertEqual(action.kind, "tap")
        self.assertEqual(action.target, "close_overlay")
        self.assertEqual(action.transition, "overlay_close_requested")

    def test_close_overlay_must_be_reobserved_before_any_back_action(self):
        controller = StateGraphController()
        context = {
            "operation": "wechat.send_text",
            "params": {"chat_name": "文件传输助手", "text": "你好"},
        }
        blocked = parse_page_observation(
            {
                "base_state": "wechat_chat",
                "overlays": ["app_dialog"],
                "confidence": 0.98,
                "stable": True,
                "page_fingerprint": "dialog-open",
                "targets": {"close_overlay": [920, 80]},
                "target_bounds": {"close_overlay": [890, 50, 950, 110]},
            }
        )

        close_action = controller.next_action(blocked, context)
        self.assertEqual(close_action.kind, "tap")
        controller.apply_action_result(close_action, context, blocked)

        # The controller must observe and verify the close result first. It
        # must never issue Android Back immediately after tapping the X.
        verification = controller.confirm_action_result(blocked, context)
        self.assertEqual(verification.kind, "wait")

        recovered = parse_page_observation(
            {
                "base_state": "wechat_chat",
                "overlays": [],
                "confidence": 0.98,
                "stable": True,
                "page_fingerprint": "dialog-closed",
                "targets": {"chat_input": [500, 820]},
            }
        )
        self.assertIsNone(controller.confirm_action_result(recovered, context))
        next_action = controller.next_action(recovered, context)
        self.assertNotEqual(next_action.kind, "back")

    def test_unknown_overlay_is_fail_closed(self):
        result = parse_page_observation(
            {
                "base_state": "wechat_chat",
                "overlays": ["mystery_sheet"],
                "confidence": 0.98,
                "stable": True,
            }
        )
        self.assertEqual(result.overlays, ("unknown_overlay",))
        self.assertEqual(result.blocking_overlays, ("unknown_overlay",))

    def test_legacy_composite_state_remains_compatible(self):
        result = parse_page_observation(
            {
                "state": "wechat_chat_keyboard",
                "confidence": 0.98,
                "stable": True,
                "keyboard_visible": True,
            }
        )
        self.assertEqual(result.base_state, "wechat_chat")
        self.assertEqual(result.overlays, ("keyboard",))
        self.assertEqual(result.state, "wechat_chat_keyboard")

    def test_overlay_and_base_page_contradiction_is_rejected(self):
        with self.assertRaisesRegex(VisionAgentError, "评论弹层"):
            parse_page_observation(
                {
                    "base_state": "wechat_chat",
                    "overlays": ["douyin_comments"],
                    "confidence": 0.98,
                    "stable": True,
                }
            )

    def test_out_of_state_target_is_dropped_before_coordinate_validation(self):
        result = parse_page_observation(
            {
                "state": "douyin_ad",
                "confidence": 1.0,
                "stable": True,
                "targets": {
                    "close_comments": {"center_x": 142, "center_y": 51},
                },
            }
        )
        self.assertEqual(result.targets, {})

    def test_live_preview_text_overrides_model_ad_guess(self):
        result = parse_page_observation(
            {
                "base_state": "douyin_ad",
                "confidence": 0.95,
                "stable": True,
                "visible_texts": ["点击进入直播间", "直播中", "讲解中", "已售10万+"],
                "live_evidence": [
                    "点击进入直播间",
                    "直播中",
                    "本地视觉:直播中徽标",
                ],
                "live_evidence_bounds": {
                    "点击进入直播间": [300, 420, 700, 500],
                    "直播中": [80, 700, 240, 750],
                },
                "ad_evidence": [],
            }
        )
        self.assertEqual(result.state, "douyin_live_preview")
        self.assertEqual(result.base_state, "douyin_live_preview")
        self.assertIn("点击进入直播间", result.live_evidence)

    def test_explicit_live_preview_discards_hallucinated_comments_overlay(self):
        result = parse_page_observation(
            {
                "base_state": "douyin_video",
                "overlays": ["douyin_comments"],
                "confidence": 0.95,
                "stable": True,
                "visible_texts": [
                    "点击进入直播间",
                    "直播中",
                    "@春野小鹿",
                    "讲解中",
                ],
                "live_evidence": ["直播中", "本地视觉:直播中徽标"],
                "live_evidence_bounds": {"直播中": [80, 700, 240, 750]},
            }
        )
        self.assertEqual(result.state, "douyin_live_preview")
        self.assertEqual(result.base_state, "douyin_live_preview")
        self.assertNotIn("douyin_comments", result.overlays)
        self.assertIn("已忽略缺少独立证据的评论弹层标签", result.reason)

    def test_local_live_preview_badge_alone_cannot_override_model_guess(self):
        result = parse_page_observation(
            {
                "base_state": "douyin_ad",
                "confidence": 0.95,
                "stable": True,
                "live_evidence": ["本地视觉:直播中徽标"],
            }
        )
        self.assertEqual(result.state, "unknown")

    def test_live_evidence_wins_when_shopping_card_looks_like_ad(self):
        result = parse_page_observation(
            {
                "base_state": "douyin_ad",
                "confidence": 0.95,
                "stable": True,
                "visible_texts": ["直播中", "点击进入直播间", "立即下载"],
                "live_evidence": [
                    "直播中",
                    "点击进入直播间",
                    "本地视觉:直播中徽标",
                ],
                "live_evidence_bounds": {
                    "直播中": [80, 700, 240, 750],
                    "点击进入直播间": [300, 420, 700, 500],
                },
                "ad_evidence": ["立即下载"],
            }
        )
        self.assertEqual(result.state, "douyin_live_preview")

    def test_hallucinated_live_words_with_boxes_but_no_local_signal_are_rejected(self):
        result = parse_page_observation(
            {
                "base_state": "douyin_live_preview",
                "confidence": 0.99,
                "stable": True,
                "live_evidence": ["直播中", "点击进入直播间"],
                "live_evidence_bounds": {
                    "直播中": [80, 700, 240, 750],
                    "点击进入直播间": [300, 420, 700, 500],
                },
                "heart_state": "unliked",
                "targets": {"heart": [880, 420], "comments": [880, 540]},
                "target_bounds": {
                    "heart": [830, 370, 930, 470],
                    "comments": [830, 490, 930, 590],
                },
            }
        )
        self.assertEqual(result.state, "unknown")
        self.assertIn("本地视觉一致", result.reason)

    def test_ad_requires_explicit_ad_evidence(self):
        unsupported = parse_page_observation(
            {
                "base_state": "douyin_ad",
                "confidence": 0.95,
                "stable": True,
                "visible_texts": ["商品名称", "已售10万+"],
            }
        )
        explicit = parse_page_observation(
            {
                "base_state": "douyin_video",
                "confidence": 0.95,
                "stable": True,
                "visible_texts": ["广告", "立即下载"],
            }
        )
        self.assertEqual(unsupported.state, "unknown")
        self.assertEqual(explicit.state, "douyin_ad")

    def test_ordinary_video_with_standard_controls_remains_video(self):
        result = parse_page_observation(
            {
                "base_state": "douyin_video",
                "confidence": 0.95,
                "stable": True,
                "heart_state": "unliked",
                "targets": {"heart": [880, 420], "comments": [880, 540]},
                "target_bounds": {
                    "heart": [830, 370, 930, 470],
                    "comments": [830, 490, 930, 590],
                },
                "visible_texts": ["作者", "作品说明"],
            }
        )
        self.assertEqual(result.state, "douyin_video")

    def test_douyin_search_tab_grid_is_not_accepted_as_home(self):
        result = parse_page_observation(
            {
                "state": "douyin_home",
                "confidence": 1.0,
                "stable": True,
                "visible_texts": ["综合", "视频", "用户", "商品", "直播"],
                "targets": {"open_search": [800, 21]},
            }
        )
        self.assertEqual(result.state, "douyin_search_results")
        self.assertNotIn("open_search", result.targets)

    def test_in_state_target_keeps_strict_coordinate_validation(self):
        with self.assertRaises(VisionAgentError):
            parse_page_observation(
                {
                    "state": "douyin_search",
                    "confidence": 1.0,
                    "stable": True,
                    "targets": {
                        "submit_search": {"center_x": 900, "center_y": 850},
                    },
                }
            )

    def test_candidate_text_requires_matching_coordinate(self):
        with self.assertRaisesRegex(VisionAgentError, "exact_candidate"):
            parse_page_observation(
                {
                    "state": "douyin_search",
                    "confidence": 0.98,
                    "stable": True,
                    "composition_text": "ji'xie'bi",
                    "candidate_text": "机械臂",
                    "targets": {},
                }
            )

    def test_observer_repairs_candidate_coordinate_once(self):
        class Provider:
            def __init__(self):
                self.calls = 0

            def status(self):
                return {"configured": True}

            def _chat(self, messages, max_tokens):
                self.calls += 1
                targets = {} if self.calls == 1 else {"exact_candidate": [150, 620]}
                return __import__("json").dumps(
                    {
                        "state": "douyin_search",
                        "confidence": 0.98,
                        "stable": True,
                        "composition_text": "ji'xie'bi",
                        "candidate_text": "机械臂",
                        "targets": targets,
                    },
                    ensure_ascii=False,
                )

        provider = Provider()
        result = DashScopePageObserver(provider).observe(
            operation="douyin.search",
            params={"keyword": "机械臂"},
            frames=[Image.new("RGB", (8, 8)) for _ in range(4)],
            controller_context={},
        )
        self.assertEqual(provider.calls, 2)
        self.assertEqual(result.targets["exact_candidate"], (150, 620))

    def test_low_confidence_stops(self):
        controller = StateGraphController()
        action = controller.next_action(
            observation("wechat_home", confidence=0.6),
            {
                "operation": "wechat.send_text",
                "params": {"chat_name": "文件传输助手", "text": "你好"},
            },
        )
        self.assertEqual(action.kind, "fail")

    def test_unknown_page_back_must_change_observed_state(self):
        controller = StateGraphController()
        context = {
            "operation": "wechat.send_text",
            "params": {"chat_name": "文件传输助手", "text": "你好"},
        }
        unknown = observation("unknown", page_fingerprint="same-unknown")
        action = controller.next_action(unknown, context)
        self.assertEqual(action.kind, "back")
        controller.apply_action_result(action, context, unknown)
        self.assertEqual(controller.confirm_action_result(unknown, context).kind, "wait")
        failure = controller.confirm_action_result(unknown, context)
        self.assertEqual(failure.kind, "fail")
        self.assertIn("连续两次", failure.reason)

    def test_recovery_counter_resets_after_known_page(self):
        controller = StateGraphController()
        context = {
            "operation": "wechat.send_text",
            "params": {"chat_name": "文件传输助手", "text": "你好"},
            "recovery_actions": 2,
        }
        action = controller.next_action(
            observation("wechat_home", targets={"open_search": (900, 80)}),
            context,
        )
        self.assertEqual(action.target, "open_search")
        self.assertNotIn("recovery_actions", context)

    def test_recovery_has_verified_action_limit(self):
        controller = StateGraphController()
        context = {
            "operation": "wechat.send_text",
            "params": {"chat_name": "文件传输助手", "text": "你好"},
            "recovery_actions": 3,
        }
        action = controller.next_action(observation("unknown"), context)
        self.assertEqual(action.kind, "fail")
        self.assertIn("3个", action.reason)

    def test_app_launch_records_controller_owned_start_evidence(self):
        controller = StateGraphController()
        context = {
            "operation": "douyin.search",
            "params": {"keyword": "测试"},
        }
        home = observation(
            "android_home",
            targets={"douyin_icon": (800, 700)},
        )
        action = controller.next_action(home, context)
        self.assertEqual(action.target, "douyin_icon")
        self.assertEqual(action.transition, "app_launch_requested")
        controller.apply_action_result(action, context, home)
        opened = observation("douyin_search", keyboard_visible=True)
        self.assertIsNone(controller.confirm_action_result(opened, context))
        self.assertTrue(context["app_opened_by_task"])

    def test_task_launched_app_normalizes_resumed_search_page_once(self):
        controller = StateGraphController()
        context = {
            "operation": "douyin.search",
            "params": {"keyword": "测试"},
            "app_opened_by_task": True,
        }
        resumed = observation(
            "douyin_search",
            keyboard_visible=True,
            input_scope="unverified",
            input_focused=True,
        )
        first = controller.next_action(resumed, context)
        self.assertEqual(first.kind, "back")
        self.assertEqual(first.transition, "startup_normalize_back_requested")
        controller.apply_action_result(first, context, resumed)
        home = observation(
            "douyin_home",
            page_fingerprint="douyin-home",
            targets={"open_search": (900, 70)},
        )
        self.assertIsNone(controller.confirm_action_result(home, context))
        self.assertTrue(context["startup_normalize_back_requested"])
        next_action = controller.next_action(
            home,
            context,
        )
        self.assertEqual(next_action.target, "open_search")

    def test_task_launch_normalization_never_repeats_back_on_search_page(self):
        controller = StateGraphController()
        context = {
            "operation": "douyin.search",
            "params": {"keyword": "测试"},
            "app_opened_by_task": True,
            "startup_normalize_back_requested": True,
        }
        resumed = observation(
            "douyin_search",
            keyboard_visible=True,
            input_scope="unverified",
        )
        action = controller.next_action(resumed, context)
        self.assertEqual(action.kind, "fail")
        self.assertNotEqual(action.kind, "back")
        self.assertIn("仍停在旧搜索页", action.reason)

    def test_flat_qwerty_anchors_are_normalized(self):
        result = parse_page_observation(
            {
                "state": "douyin_search",
                "confidence": 1.0,
                "stable": True,
                "keyboard_visible": True,
                "keyboard_layout": {
                    "type": "qwerty",
                    "q": [124, 697],
                    "p": [885, 697],
                    "a": [170, 769],
                    "l": [843, 769],
                    "z": [216, 841],
                    "m": [797, 841],
                    "backspace": [874, 841],
                },
            }
        )
        self.assertEqual(result.keyboard_layout["type"], "qwerty")
        self.assertEqual(result.keyboard_layout["anchors"]["q"], [124, 697])
        self.assertEqual(result.keyboard_layout["anchors"]["backspace"], [874, 841])

    def test_qwerty_anchor_boxes_are_normalized_to_centers(self):
        result = parse_page_observation(
            {
                "state": "douyin_search",
                "confidence": 1.0,
                "stable": True,
                "keyboard_visible": True,
                "keyboard_layout": {
                    "type": "qwerty",
                    "q": [124, 697, 154, 717],
                    "p": [874, 697, 904, 717],
                    "a": [124, 767, 154, 787],
                    "l": [834, 767, 864, 787],
                    "z": [124, 837, 154, 857],
                    "m": [714, 837, 744, 857],
                    "backspace": [874, 837, 904, 857],
                },
            }
        )
        self.assertEqual(result.keyboard_layout["anchors"]["q"], [139, 707])
        self.assertEqual(
            result.keyboard_layout["anchors"]["backspace"],
            [889, 847],
        )

    def test_oversized_keyboard_anchor_box_is_rejected(self):
        with self.assertRaises(VisionAgentError):
            parse_page_observation(
                {
                    "state": "douyin_search",
                    "confidence": 1.0,
                    "stable": True,
                    "keyboard_visible": True,
                    "keyboard_layout": {
                        "type": "generic",
                        "backspace": [100, 600, 500, 900],
                    },
                }
            )

    def test_active_input_requires_focus_and_own_bounds(self):
        with self.assertRaises(VisionAgentError):
            parse_page_observation(
                {
                    "state": "wechat_chat_keyboard",
                    "confidence": 1.0,
                    "stable": True,
                    "input_scope": "active_input",
                    "input_text": "",
                    "input_is_empty": True,
                    "keyboard_visible": True,
                }
            )

    def test_active_input_bounds_must_be_in_page_input_region(self):
        observed = parse_page_observation(
            {
                "state": "wechat_chat_keyboard",
                "confidence": 1.0,
                "stable": True,
                "input_scope": "active_input",
                "input_text": "你好",
                "input_is_empty": False,
                "input_focused": True,
                # A chat bubble near the top, not the bottom input field.
                "input_bounds": [500, 100, 850, 240],
                "keyboard_visible": True,
            }
        )
        self.assertEqual(observed.input_scope, "unverified")
        self.assertIsNone(observed.input_text)
        self.assertIsNone(observed.input_is_empty)
        self.assertIsNone(observed.input_focused)
        self.assertIsNone(observed.input_bounds)

    def test_invalid_input_roi_does_not_weaken_visible_wechat_input_gate(self):
        observed = parse_page_observation(
            {
                "state": "wechat_chat_keyboard",
                "confidence": 1.0,
                "stable": True,
                "page_title": "文件传输助手",
                "input_scope": "active_input",
                "input_text": "你好",
                "input_is_empty": False,
                "input_focused": True,
                "input_bounds": [500, 100, 850, 240],
                "keyboard_visible": True,
            }
        )
        action = StateGraphController().next_action(
            observed,
            {
                "operation": "wechat.send_text",
                "params": {"chat_name": "文件传输助手", "text": "你好"},
            },
        )
        self.assertEqual(action.kind, "fail")
        self.assertIn("当前输入框ROI", action.reason)


class StateGraphControllerTests(unittest.TestCase):
    def setUp(self):
        self.controller = StateGraphController()
        self.context = {
            "operation": "wechat.send_text",
            "params": {"chat_name": "文件传输助手", "text": "你好"},
        }

    def test_current_chat_state_skips_all_prior_navigation(self):
        self.context["fields"] = {
            "wechat_message": {
                "started": True,
                "retypes": 0,
                "awaiting_empty": False,
            }
        }
        action = self.controller.next_action(
            observation(
                "wechat_chat_keyboard",
                page_title="文件传输助手",
                input_text="你好",
                input_is_empty=False,
                keyboard_visible=True,
                targets={"send": (900, 850)},
            ),
            self.context,
        )
        self.assertEqual(action.kind, "tap")
        self.assertEqual(action.target, "send")

    def test_preexisting_exact_target_text_is_not_submitted(self):
        action = self.controller.next_action(
            observation(
                "wechat_chat_keyboard",
                page_title="文件传输助手",
                input_text="你好",
                input_is_empty=False,
                keyboard_visible=True,
                targets={"send": (900, 850)},
            ),
            self.context,
        )
        self.assertEqual(action.kind, "fail")
        self.assertIn("不是由本任务输入", action.reason)

    def test_preexisting_target_prefix_is_not_appended(self):
        action = self.controller.next_action(
            observation(
                "wechat_chat_keyboard",
                page_title="文件传输助手",
                input_text="你",
                input_is_empty=False,
                keyboard_visible=True,
                keyboard_layout={"type": "qwerty", "anchors": {"q": [100, 700]}},
            ),
            self.context,
        )
        self.assertEqual(action.kind, "fail")
        self.assertIn("接管", action.reason)

    def test_wrong_task_input_uses_exact_backspace_count(self):
        self.context["fields"] = {
            "wechat_message": {
                "started": True,
                "retypes": 0,
                "awaiting_empty": False,
            }
        }
        action = self.controller.next_action(
            observation(
                "wechat_chat_keyboard",
                page_title="文件传输助手",
                input_text="你号",
                input_is_empty=False,
                keyboard_visible=True,
                keyboard_layout={"type": "generic", "anchors": {"backspace": [900, 800]}},
            ),
            self.context,
        )
        self.assertEqual(action.kind, "clear_text")
        self.assertEqual(action.delete_count, 2)

    def test_input_field_cannot_change_during_same_text_session(self):
        self.context["fields"] = {
            "wechat_message": {
                "started": True,
                "retypes": 0,
                "awaiting_empty": False,
                "input_bounds": [80, 470, 850, 650],
            }
        }
        action = self.controller.next_action(
            observation(
                "wechat_chat_keyboard",
                page_title="文件传输助手",
                input_text="你",
                input_is_empty=False,
                input_bounds=(80, 700, 850, 800),
                keyboard_visible=True,
                keyboard_layout={"type": "generic", "anchors": {"backspace": [900, 800]}},
            ),
            self.context,
        )
        self.assertEqual(action.kind, "fail")
        self.assertIn("另一个输入框", action.reason)

    def test_ascii_mutation_requires_visible_keyboard(self):
        context = {
            "operation": "wechat.send_text",
            "params": {"chat_name": "文件传输助手", "text": "hello"},
        }
        action = self.controller._text_action(
            observation(
                "wechat_chat_keyboard",
                page_title="文件传输助手",
                input_text="",
                input_is_empty=True,
                keyboard_visible=False,
            ),
            "hello",
            context,
        )
        self.assertEqual(action.kind, "fail")
        self.assertIn("ASCII", action.reason)

    def test_confirmed_empty_resets_stale_attempt_before_retyping(self):
        self.context["fields"] = {
            "wechat_message": {
                "started": True,
                "retypes": 1,
                "awaiting_empty": True,
                "confirmed_text": "错误",
                "pending_segment": "错误",
                "candidate_pending": True,
                "candidate_expected_text": "错误",
            }
        }
        action = self.controller.next_action(
            observation(
                "wechat_chat_keyboard",
                page_title="文件传输助手",
                input_text="",
                input_is_empty=True,
                keyboard_visible=True,
                keyboard_layout={
                    "type": "qwerty",
                    "anchors": {
                        "q": [100, 700], "p": [900, 700],
                        "a": [150, 770], "l": [850, 770],
                        "z": [240, 840], "m": [760, 840],
                        "backspace": [900, 840],
                    },
                },
            ),
            self.context,
        )
        self.assertEqual(action.kind, "type_pinyin")
        self.assertEqual(action.pinyin, "nihao")
        field = self.context["fields"]["wechat_message"]
        self.assertEqual(field["confirmed_text"], "")
        self.assertFalse(field["candidate_pending"])

    def test_existing_user_text_is_not_deleted(self):
        action = self.controller.next_action(
            observation(
                "wechat_chat_keyboard",
                page_title="文件传输助手",
                input_text="用户草稿",
                input_is_empty=False,
                keyboard_visible=True,
            ),
            self.context,
        )
        self.assertEqual(action.kind, "fail")
        self.assertIn("禁止自动删除", action.reason)

    def test_send_is_verified_on_next_observation(self):
        self.context["send_clicked"] = True
        action = self.controller.next_action(
            observation(
                "wechat_chat",
                page_title="文件传输助手",
                input_text="",
                input_is_empty=True,
                sent_message_visible=True,
            ),
            self.context,
        )
        self.assertEqual(action.kind, "finish")

    def test_chinese_pinyin_is_generated_locally(self):
        action = self.controller.next_action(
            observation(
                "wechat_chat_keyboard",
                page_title="文件传输助手",
                input_text="",
                input_is_empty=True,
                keyboard_visible=True,
                keyboard_layout={
                    "type": "qwerty",
                    "anchors": {
                        "q": [100, 700], "p": [900, 700],
                        "a": [150, 770], "l": [850, 770],
                        "z": [240, 840], "m": [760, 840],
                        "backspace": [900, 840],
                    },
                },
            ),
            self.context,
        )
        self.assertEqual(action.kind, "type_pinyin")
        self.assertEqual(action.pinyin, "nihao")

    def test_exact_candidate_can_be_selected_while_final_input_is_uncommitted(self):
        self.context["fields"] = {
            "wechat_message": {
                "started": True,
                "retypes": 0,
                "awaiting_empty": False,
                "pending_segment": "你好",
            }
        }
        action = self.controller.next_action(
            observation(
                "wechat_chat_keyboard",
                page_title="文件传输助手",
                input_text=None,
                input_is_empty=None,
                composition_text="nihao",
                candidate_text="你好",
                keyboard_visible=True,
                targets={"exact_candidate": (180, 620)},
            ),
            self.context,
        )
        self.assertEqual(action.kind, "tap")
        self.assertEqual(action.target, "exact_candidate")

    def test_exact_candidate_is_not_clicked_twice_after_text_is_committed(self):
        context = {
            "operation": "douyin.search",
            "params": {"keyword": "机械臂"},
            "fields": {
                "douyin_search": {
                    "started": True,
                    "retypes": 0,
                    "awaiting_empty": False,
                    "pending_segment": "机械臂",
                    "confirmed_text": "",
                }
            },
        }
        first = self.controller.next_action(
            observation(
                "douyin_search",
                input_text=None,
                input_is_empty=None,
                composition_text="jixiebi",
                candidate_text="机械臂",
                keyboard_visible=True,
                targets={"exact_candidate": (180, 620)},
            ),
            context,
        )
        self.assertEqual(first.target, "exact_candidate")
        second = self.controller.next_action(
            observation(
                "douyin_search",
                input_text="机械臂",
                input_is_empty=False,
                composition_text="jixiebi",
                candidate_text="机械臂",
                keyboard_visible=True,
                targets={"exact_candidate": (180, 620), "submit_search": (900, 100)},
            ),
            context,
        )
        self.assertEqual(second.kind, "tap")
        self.assertEqual(second.target, "submit_search")
        self.assertEqual(second.transition, "search_submitted")

    def test_candidate_mismatch_after_click_fails_without_second_tap(self):
        context = {
            "operation": "douyin.search",
            "params": {"keyword": "机械臂"},
            "fields": {
                "douyin_search": {
                    "started": True,
                    "retypes": 0,
                    "awaiting_empty": False,
                    "pending_segment": "机械臂",
                    "confirmed_text": "",
                    "candidate_pending": True,
                    "candidate_expected_text": "机械臂",
                }
            },
        }
        action = self.controller.next_action(
            observation(
                "douyin_search",
                input_text="机械被",
                input_is_empty=False,
                composition_text="jixiebi",
                candidate_text="机械臂",
                keyboard_visible=True,
                targets={"exact_candidate": (180, 620)},
            ),
            context,
        )
        self.assertEqual(action.kind, "fail")
        self.assertIn("禁止重复点击", action.reason)

    def test_like_requires_post_tap_red_heart_observation(self):
        context = {
            "operation": "douyin.batch_interact",
            "params": {
                "keyword": None,
                "target_count": 1,
                "like": True,
                "comment": False,
                "comment_text": None,
            },
        }
        before = observation(
            "douyin_video",
            page_fingerprint="author|caption",
            heart_state="unliked",
            targets={"heart": (900, 430)},
        )
        first = self.controller.next_action(before, context)
        self.assertEqual(first.kind, "tap")
        self.controller.apply_action_result(first, context, before)
        self.assertFalse(context.get("like_pending", False))
        after = observation(
            "douyin_video",
            page_fingerprint="author|caption",
            heart_state="liked",
        )
        self.assertIsNone(self.controller.confirm_action_result(after, context))
        second = self.controller.next_action(after, context)
        self.assertEqual(second.kind, "finish")

    def test_search_cannot_finish_on_normal_video_before_submit(self):
        context = {
            "operation": "douyin.search",
            "params": {"keyword": "机械臂"},
        }
        action = self.controller.next_action(
            observation(
                "douyin_video",
                page_fingerprint="author|caption",
                targets={"open_search": (920, 80)},
            ),
            context,
        )
        self.assertEqual(action.kind, "tap")
        self.assertEqual(action.target, "open_search")
        self.assertNotEqual(action.kind, "finish")

    def test_stale_search_result_reopens_search_instead_of_finishing(self):
        context = {
            "operation": "douyin.search",
            "params": {"keyword": "机械臂"},
        }
        stale_result = observation(
            "douyin_search_results",
            targets={},
        )
        action = self.controller.next_action(stale_result, context)
        self.assertEqual(action.kind, "back")
        self.assertEqual(action.transition, "search_recovery_back_requested")

        self.controller.apply_action_result(action, context, stale_result)
        waiting = self.controller.confirm_action_result(stale_result, context)
        self.assertEqual(waiting.kind, "wait")
        repeated = self.controller.confirm_action_result(stale_result, context)
        self.assertEqual(repeated.kind, "fail")
        self.assertIn("未达到预期结果", repeated.reason)

    def test_stale_search_result_back_may_resume_from_home(self):
        context = {
            "operation": "douyin.search",
            "params": {"keyword": "机械臂"},
            "search_recovery_back_requested": True,
        }
        action = self.controller.next_action(
            observation(
                "douyin_home",
                page_fingerprint="douyin-home",
                targets={"open_search": (860, 20)},
            ),
            context,
        )
        self.assertEqual(action.kind, "tap")
        self.assertEqual(action.target, "open_search")

    def test_search_finish_requires_submit_and_verified_result(self):
        context = {
            "operation": "douyin.search",
            "params": {"keyword": "机械臂"},
            "search_open_requested": True,
            "fields": {
                "douyin_search": {
                    "started": True,
                    "retypes": 0,
                    "awaiting_empty": False,
                    "candidate_pending": False,
                }
            },
        }
        before = observation(
            "douyin_search",
            input_text="机械臂",
            input_is_empty=False,
            targets={"submit_search": (900, 100)},
        )
        submit = self.controller.next_action(before, context)
        self.assertEqual(submit.kind, "tap")
        self.assertEqual(submit.transition, "search_submitted")
        self.controller.apply_action_result(submit, context, before)
        result_page = observation(
            "douyin_search_results",
            search_results_relevant=True,
            search_result_evidence=["机械臂自动抓取演示", "工业机械臂控制教程"],
        )
        self.assertIsNone(self.controller.confirm_action_result(result_page, context))
        finish = self.controller.next_action(result_page, context)
        self.assertEqual(finish.kind, "finish")
        self.controller.validate_finish(
            observation("douyin_search_results"),
            context,
        )

    def test_history_keyword_is_not_accepted_as_search_input(self):
        context = {
            "operation": "douyin.search",
            "params": {"keyword": "机械臂"},
            "search_open_requested": True,
        }
        action = self.controller.next_action(
            observation(
                "douyin_search",
                input_text="机械臂",
                input_is_empty=False,
                input_scope="page_text",
                keyboard_visible=True,
                keyboard_layout={"type": "qwerty", "anchors": {"q": [100, 700]}},
                targets={"submit_search": (900, 900)},
                visible_texts=["历史记录", "机械臂"],
            ),
            context,
        )
        self.assertEqual(action.kind, "type_pinyin")
        self.assertNotEqual(action.target, "submit_search")

    def test_hidden_search_field_requires_exact_candidate_causal_chain(self):
        context = {
            "operation": "douyin.search",
            "params": {"keyword": "机械臂"},
            "search_open_requested": True,
        }
        layout = {"type": "qwerty", "anchors": {"q": [100, 700]}}
        typed = self.controller.next_action(
            observation(
                "douyin_search",
                input_scope="unverified",
                keyboard_visible=True,
                keyboard_layout=layout,
            ),
            context,
        )
        self.assertEqual(typed.kind, "type_pinyin")
        candidate = self.controller.next_action(
            observation(
                "douyin_search",
                input_scope="unverified",
                composition_text="jixiebi",
                candidate_text="机械臂",
                keyboard_visible=True,
                keyboard_layout=layout,
                targets={"exact_candidate": (180, 620)},
            ),
            context,
        )
        self.assertEqual(candidate.target, "exact_candidate")
        submit = self.controller.next_action(
            observation(
                "douyin_search",
                input_scope="unverified",
                composition_text=None,
                candidate_text=None,
                keyboard_visible=True,
                keyboard_layout=layout,
                targets={"submit_search": (900, 900)},
            ),
            context,
        )
        self.assertEqual(submit.target, "submit_search")

    def test_unrelated_search_results_cannot_finish(self):
        context = {
            "operation": "douyin.search",
            "params": {"keyword": "机械臂"},
            "search_submitted": True,
        }
        action = self.controller.next_action(
            observation(
                "douyin_search_results",
                search_results_relevant=False,
                search_result_evidence=["F1匈牙利大奖赛", "维斯塔潘"],
            ),
            context,
        )
        self.assertEqual(action.kind, "fail")
        self.assertIn("机械臂", action.reason)

    def test_result_page_exact_query_box_can_finish(self):
        context = {
            "operation": "douyin.search",
            "params": {"keyword": "机械臂"},
            "search_submitted": True,
        }
        action = self.controller.next_action(
            observation(
                "douyin_search_results",
                search_query_text="机械臂",
                search_query_verified=True,
            ),
            context,
        )
        self.assertEqual(action.kind, "finish")
        self.assertEqual(context["verified_search_keyword"], "机械臂")

    def test_keyword_batch_cannot_operate_recommendation_before_search(self):
        context = {
            "operation": "douyin.batch_interact",
            "params": {
                "keyword": "机械臂",
                "target_count": 1,
                "like": True,
                "comment": False,
                "comment_text": None,
            },
        }
        action = self.controller.next_action(
            observation(
                "douyin_video",
                page_fingerprint="recommendation",
                heart_state="unliked",
                targets={"open_search": (920, 80), "heart": (900, 430)},
            ),
            context,
        )
        self.assertEqual(action.target, "open_search")

    def test_same_video_after_swipe_is_not_counted_twice(self):
        context = {
            "operation": "douyin.batch_interact",
            "params": {
                "keyword": None,
                "target_count": 2,
                "like": False,
                "comment": False,
                "comment_text": None,
            },
        }
        before = observation("douyin_video", page_fingerprint="video-a")
        first = self.controller.next_action(before, context)
        self.assertEqual(first.kind, "swipe_up")
        self.controller.apply_action_result(first, context, before)
        same = observation("douyin_video", page_fingerprint="video-a")
        second = self.controller.confirm_action_result(same, context)
        self.assertEqual(second.kind, "wait")
        third = self.controller.confirm_action_result(same, context)
        self.assertEqual(third.kind, "fail")
        self.assertEqual(context["completed"], 1)

    def test_comment_is_not_complete_until_comments_page_closes(self):
        context = {
            "operation": "douyin.batch_interact",
            "params": {
                "keyword": None,
                "target_count": 1,
                "like": False,
                "comment": True,
                "comment_text": "1",
            },
            "active_video": "video-a",
            "like_done": True,
            "comment_done": False,
            "comment_submitted": True,
        }
        comments_page = observation(
            "douyin_comments",
            comment_sent_visible=True,
            targets={"close_comments": (950, 150)},
        )
        close = self.controller.next_action(comments_page, context)
        self.controller.apply_action_result(close, context, comments_page)
        self.assertFalse(context["comment_done"])
        waiting = self.controller.confirm_action_result(
            observation("douyin_comments", comment_sent_visible=True),
            context,
        )
        self.assertEqual(waiting.kind, "wait")
        video_page = observation("douyin_video", page_fingerprint="video-a")
        self.assertIsNone(self.controller.confirm_action_result(video_page, context))
        finish = self.controller.next_action(video_page, context)
        self.assertEqual(finish.kind, "finish")
        self.assertTrue(context["comment_done"])


class WorkflowPayloadTests(unittest.TestCase):
    def test_structured_workflow_contains_no_linear_plan(self):
        payload = build_state_workflow_params(
            "wechat.send_text",
            {"chat_name": "文件传输助手", "text": "你好"},
        )
        self.assertEqual(payload["workflow_version"], "page_state_graph_v1")
        self.assertEqual(payload["model_role"], "observation_only")
        self.assertNotIn("execution_plan", payload)
        self.assertNotIn("allowed_texts", payload)


class LocalTargetResolverTests(unittest.TestCase):
    @staticmethod
    def _ocr_payload(text="搜索", left=53, top=11, width=36, height=18):
        return {
            "lines": [
                {
                    "text": text,
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height,
                    "words": [
                        {
                            "text": text,
                            "left": left,
                            "top": top,
                            "width": width,
                            "height": height,
                        }
                    ],
                }
            ]
        }

    @classmethod
    def _duplicate_search_payload(cls, *, include_bottom: bool):
        top = cls._ocr_payload(left=130, top=11)["lines"][0]
        lines = [top]
        if include_bottom:
            lines.append(cls._ocr_payload(left=130, top=860)["lines"][0])
        return {"lines": lines}

    def test_search_result_button_uses_local_ocr_not_qwen_point(self):
        resolver = LocalTargetResolver(
            ocr_recognizer=lambda *_args, **_kwargs: self._ocr_payload()
        )
        frames = [Image.new("RGB", (540, 960), "white") for _ in range(4)]
        result = resolver.resolve(
            frames=frames,
            observation=observation("douyin_search_results"),
            target="open_search",
            proposed=(800, 24),
            params={"keyword": "机械臂"},
        )
        self.assertEqual(result.method, "windows_ocr_multiframe")
        self.assertEqual(result.pixel_center, (449, 34))
        self.assertEqual(result.resolved_coordinate, (833, 35))
        self.assertNotEqual(result.resolved_coordinate, result.proposed_coordinate)

    def test_app_icon_ocr_confirms_identity_but_clicks_model_icon_box(self):
        # Full-frame OCR label center is about (413, 715), below the icon.
        # The Android-home OCR crop starts at (10, 48), hence this payload.
        resolver = LocalTargetResolver(
            ocr_recognizer=lambda *_args, **_kwargs: self._ocr_payload(
                text="抖音", left=395, top=658, width=36, height=18
            )
        )
        frames = [Image.new("RGB", (540, 960), "white") for _ in range(4)]
        result = resolver.resolve(
            frames=frames,
            observation=observation(
                "android_home",
                target_bounds={"douyin_icon": (700, 640, 850, 750)},
            ),
            target="douyin_icon",
            proposed=(800, 700),
            params={},
        )
        self.assertEqual(result.method, "qwen_box_ocr_identity_fusion")
        self.assertEqual(result.resolved_coordinate, (775, 695))
        self.assertEqual(result.pixel_center, (418, 667))
        self.assertLess(result.pixel_center[1], 700)

    def test_exact_candidate_is_allowed_only_in_verified_keyboard_band(self):
        # The OCR crop begins at y=432 on a 960px frame.  This payload places
        # the exact candidate near full-frame (95, 607), inside the model's
        # independently supplied candidate bounds.
        resolver = LocalTargetResolver(
            ocr_recognizer=lambda *_args, **_kwargs: self._ocr_payload(
                text="测试", left=70, top=165, width=50, height=20
            )
        )
        frames = [Image.new("RGB", (540, 960), "white") for _ in range(4)]
        result = resolver.resolve(
            frames=frames,
            observation=observation(
                "douyin_search",
                candidate_text="测试",
                target_bounds={"exact_candidate": (120, 590, 240, 680)},
            ),
            target="exact_candidate",
            proposed=(180, 635),
            params={"keyword": "测试"},
        )
        self.assertEqual(result.method, "windows_ocr_multiframe")
        self.assertEqual(result.pixel_center, (95, 607))
        self.assertEqual(result.resolved_coordinate, (176, 633))

    def test_exact_candidate_remains_forbidden_outside_keyboard_states(self):
        resolver = LocalTargetResolver(
            ocr_recognizer=lambda *_args, **_kwargs: self._ocr_payload(text="测试")
        )
        frames = [Image.new("RGB", (540, 960), "white") for _ in range(4)]
        with self.assertRaisesRegex(VisionAgentError, "尚无页面专属点击区域"):
            resolver.resolve(
                frames=frames,
                observation=observation(
                    "douyin_video",
                    candidate_text="测试",
                    target_bounds={"exact_candidate": (120, 590, 240, 680)},
                ),
                target="exact_candidate",
                proposed=(180, 635),
                params={"keyword": "测试"},
            )

    def test_exact_candidate_uses_guarded_model_box_when_camera_ocr_cannot_read_chinese(self):
        resolver = LocalTargetResolver(
            ocr_recognizer=lambda *_args, **_kwargs: {"lines": []}
        )
        frames = [Image.new("RGB", (540, 960), "white") for _ in range(4)]
        result = resolver.resolve(
            frames=frames,
            observation=observation(
                "douyin_search",
                confidence=0.95,
                stable=True,
                candidate_text="测试",
                visible_texts=["ce'shi", "测试", "侧室"],
                target_bounds={"exact_candidate": (100, 625, 220, 650)},
            ),
            target="exact_candidate",
            proposed=(180, 635),
            params={"keyword": "测试", "_expected_candidate_text": "测试"},
        )
        self.assertEqual(result.method, "qwen_multiframe_exact_text_box_guarded")
        self.assertEqual(result.resolved_coordinate, (160, 637))

    def test_candidate_box_fallback_requires_controller_validated_expected_text(self):
        resolver = LocalTargetResolver(
            ocr_recognizer=lambda *_args, **_kwargs: {"lines": []}
        )
        frames = [Image.new("RGB", (540, 960), "white") for _ in range(4)]
        with self.assertRaisesRegex(VisionAgentError, "未达到2帧一致性要求"):
            resolver.resolve(
                frames=frames,
                observation=observation(
                    "douyin_search",
                    confidence=0.95,
                    stable=True,
                    candidate_text="测试",
                    visible_texts=["测试"],
                    target_bounds={"exact_candidate": (100, 625, 220, 650)},
                ),
                target="exact_candidate",
                proposed=(180, 635),
                params={"keyword": "测试"},
            )

    def test_search_result_clear_uses_submit_label_as_local_anchor(self):
        resolver = LocalTargetResolver(
            ocr_recognizer=lambda *_args, **_kwargs: self._ocr_payload(left=80)
        )
        frames = [Image.new("RGB", (540, 960), "white") for _ in range(4)]
        result = resolver.resolve(
            frames=frames,
            observation=observation("douyin_search_results"),
            target="clear_search",
            proposed=(725, 22),
            params={"keyword": "机械臂"},
        )
        self.assertEqual(result.method, "windows_ocr_multiframe")
        self.assertEqual(result.pixel_center, (391, 20))
        self.assertEqual(result.samples, 4)

    def test_duplicate_ocr_label_prefers_match_inside_model_bounds(self):
        resolver = LocalTargetResolver(
            ocr_recognizer=lambda *_args, **_kwargs: self._duplicate_search_payload(
                include_bottom=True
            )
        )
        frames = [Image.new("RGB", (540, 960), "white") for _ in range(4)]
        result = resolver.resolve(
            frames=frames,
            observation=observation(
                "douyin_search",
                target_bounds={"submit_search": (800, 880, 920, 940)},
            ),
            target="submit_search",
            proposed=(860, 910),
            params={"keyword": "测试"},
        )
        self.assertEqual(result.method, "windows_ocr_multiframe")
        self.assertGreater(result.resolved_coordinate[1], 850)

    def test_distant_duplicate_ocr_cannot_override_model_target_box(self):
        resolver = LocalTargetResolver(
            ocr_recognizer=lambda *_args, **_kwargs: self._duplicate_search_payload(
                include_bottom=False
            )
        )
        frames = [Image.new("RGB", (540, 960), "white") for _ in range(4)]
        result = resolver.resolve(
            frames=frames,
            observation=observation(
                "douyin_search",
                target_bounds={"submit_search": (800, 880, 920, 940)},
            ),
            target="submit_search",
            proposed=(860, 910),
            params={"keyword": "测试"},
        )
        self.assertEqual(result.method, "qwen_box_ocr_spatial_fallback")
        self.assertEqual(result.resolved_coordinate, (860, 910))

    def test_ocr_target_without_multiframe_consensus_is_rejected(self):
        payloads = iter([self._ocr_payload(), {"lines": []}, {"lines": []}, {"lines": []}])
        resolver = LocalTargetResolver(
            ocr_recognizer=lambda *_args, **_kwargs: next(payloads)
        )
        frames = [Image.new("RGB", (540, 960), "white") for _ in range(4)]
        with self.assertRaisesRegex(VisionAgentError, "未达到2帧一致性要求"):
            resolver.resolve(
                frames=frames,
                observation=observation("douyin_search_results"),
                target="open_search",
                proposed=(800, 24),
                params={"keyword": "机械臂"},
            )

    def test_target_outside_page_region_is_rejected(self):
        resolver = LocalTargetResolver()
        frames = [Image.new("RGB", (540, 960), "white") for _ in range(4)]
        with self.assertRaisesRegex(VisionAgentError, "不在wechat_chat允许区域"):
            resolver.resolve(
                frames=frames,
                observation=observation("wechat_chat"),
                target="chat_input",
                proposed=(300, 700),
                params={},
            )

    def test_model_only_target_requires_clickable_bounds(self):
        resolver = LocalTargetResolver()
        frames = [Image.new("RGB", (540, 960), "white") for _ in range(4)]
        with self.assertRaisesRegex(VisionAgentError, "没有可点击边界框"):
            resolver.resolve(
                frames=frames,
                observation=observation("douyin_video"),
                target="comments",
                proposed=(850, 600),
                params={},
            )

    def test_model_box_center_replaces_off_center_hint(self):
        resolver = LocalTargetResolver()
        frames = [Image.new("RGB", (540, 960), "white") for _ in range(4)]
        result = resolver.resolve(
            frames=frames,
            observation=observation(
                "douyin_video",
                target_bounds={"comments": (800, 560, 900, 640)},
            ),
            target="comments",
            proposed=(820, 580),
            params={},
        )
        self.assertEqual(result.method, "qwen_box_region_guarded")
        self.assertEqual(result.resolved_coordinate, (850, 600))

    def test_model_point_must_be_inside_its_own_bounds(self):
        resolver = LocalTargetResolver()
        frames = [Image.new("RGB", (540, 960), "white") for _ in range(4)]
        with self.assertRaisesRegex(VisionAgentError, "中心不在其边界框"):
            resolver.resolve(
                frames=frames,
                observation=observation(
                    "douyin_video",
                    target_bounds={"comments": (800, 560, 900, 640)},
                ),
                target="comments",
                proposed=(750, 600),
                params={},
            )

    def test_model_bounds_must_remain_inside_page_region(self):
        resolver = LocalTargetResolver()
        frames = [Image.new("RGB", (540, 960), "white") for _ in range(4)]
        with self.assertRaisesRegex(VisionAgentError, "边界框越出"):
            resolver.resolve(
                frames=frames,
                observation=observation(
                    "douyin_video",
                    target_bounds={"comments": (600, 560, 900, 640)},
                ),
                target="comments",
                proposed=(850, 600),
                params={},
            )

    def test_local_ocr_overrides_bad_model_hint_before_region_guard(self):
        resolver = LocalTargetResolver(
            ocr_recognizer=lambda *_args, **_kwargs: self._ocr_payload(left=80)
        )
        frames = [Image.new("RGB", (540, 960), "white") for _ in range(4)]
        result = resolver.resolve(
            frames=frames,
            observation=observation("douyin_search_results"),
            target="clear_search",
            proposed=(100, 800),
            params={"keyword": "机械臂"},
        )
        self.assertEqual(result.pixel_center, (391, 20))

    def test_heart_uses_local_detector_center(self):
        resolver = LocalTargetResolver(
            heart_detector=lambda _frame: SimpleNamespace(
                center=(468, 305), state="unliked"
            )
        )
        frames = [Image.new("RGB", (540, 960), "black") for _ in range(4)]
        result = resolver.resolve(
            frames=frames,
            observation=observation("douyin_video"),
            target="heart",
            proposed=(820, 350),
            params={"like": True},
        )
        self.assertEqual(result.method, "local_heart_multiframe")
        self.assertEqual(result.pixel_center, (468, 305))
        self.assertEqual(result.resolved_coordinate, (868, 318))


if __name__ == "__main__":
    unittest.main()
