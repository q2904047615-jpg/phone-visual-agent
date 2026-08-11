from __future__ import annotations

import json
import unittest

from PIL import Image, ImageDraw

from observation_images import (
    ObservationRoi,
    build_overview,
    build_roi,
    map_roi_bounds_to_full,
    map_roi_point_to_full,
    measure_local_stability,
)
from state_controller import DashScopePageObserver
from vision_agent import VisionAgentError


def patterned_frame() -> Image.Image:
    image = Image.new("RGB", (540, 960), "#eee6db")
    draw = ImageDraw.Draw(image)
    for y in range(0, 960, 48):
        draw.rectangle((0, y, 540, min(y + 23, 959)), fill=(y % 255, 70, 130))
    for x in range(20, 520, 72):
        draw.ellipse((x, 220, x + 45, 265), fill="#13a87a")
    return image


class FakeProvider:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def status(self) -> dict:
        return {"configured": True}

    def _chat(self, messages, *, max_tokens):
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        if not self.responses:
            raise AssertionError("模型被额外调用")
        return json.dumps(self.responses.pop(0), ensure_ascii=False)


class ObservationImageTests(unittest.TestCase):
    def test_overview_and_roi_stay_inside_payload_budget(self) -> None:
        image = patterned_frame()
        overview = build_overview(image)
        roi = build_roi(
            image,
            ObservationRoi("right_actions", (580, 60, 1000, 880), "测试"),
        )
        self.assertLessEqual(overview.jpeg_bytes, 28000)
        self.assertLessEqual(roi.jpeg_bytes, 24000)
        self.assertEqual(overview.width, 320)
        # The ROI covers only 42% of the original width. Even when its encoded
        # width equals the overview width, it carries substantially more pixels
        # per unit of the original screen.
        roi_screen_fraction = (1000 - 580) / 1000
        self.assertGreater(roi.width / roi_screen_fraction, overview.width)

    def test_roi_coordinates_map_back_to_full_frame(self) -> None:
        bounds = (580, 60, 1000, 880)
        self.assertEqual(map_roi_point_to_full((500, 500), bounds), (790, 470))
        self.assertEqual(
            map_roi_bounds_to_full((200, 300, 800, 700), bounds),
            (664, 306, 916, 634),
        )

    def test_local_stability_uses_all_four_original_frames(self) -> None:
        frame = patterned_frame()
        stable = measure_local_stability([frame.copy() for _ in range(4)])
        self.assertTrue(stable.stable)
        self.assertEqual(stable.frame_count, 4)

        changed = Image.new("RGB", frame.size, "white")
        unstable = measure_local_stability([frame, changed, frame, changed])
        self.assertFalse(unstable.stable)

    def test_overview_only_when_required_evidence_is_present(self) -> None:
        provider = FakeProvider(
            [
                {
                    "base_state": "android_home",
                    "confidence": 0.94,
                    "targets": {"douyin_icon": [720, 650]},
                    "target_bounds": {"douyin_icon": [660, 590, 780, 710]},
                }
            ]
        )
        observer = DashScopePageObserver(provider)
        result = observer.observe(
            operation="douyin.search",
            params={"keyword": "测试"},
            frames=[patterned_frame() for _ in range(4)],
            controller_context={},
        )
        self.assertEqual(result.state, "android_home")
        self.assertEqual(len(provider.calls), 1)
        diagnostics = observer.last_observation_diagnostics
        self.assertEqual(diagnostics["model_calls"], 1)
        self.assertEqual([item["role"] for item in diagnostics["images"]], ["overview"])
        self.assertLessEqual(diagnostics["images"][0]["jpeg_bytes"], 28000)

    def test_missing_target_requests_roi_and_maps_local_geometry(self) -> None:
        provider = FakeProvider(
            [
                {
                    "base_state": "android_home",
                    "confidence": 0.92,
                    "targets": {},
                    "target_bounds": {},
                },
                {
                    "confidence": 0.91,
                    "targets": {"douyin_icon": [500, 500]},
                    "target_bounds": {"douyin_icon": [400, 400, 600, 600]},
                },
            ]
        )
        observer = DashScopePageObserver(provider)
        result = observer.observe(
            operation="douyin.search",
            params={"keyword": "测试"},
            frames=[patterned_frame() for _ in range(4)],
            controller_context={},
        )
        self.assertEqual(result.targets["douyin_icon"], (500, 480))
        self.assertEqual(result.target_bounds["douyin_icon"], (400, 390, 600, 570))
        self.assertEqual(len(provider.calls), 2)
        diagnostics = observer.last_observation_diagnostics
        self.assertEqual(diagnostics["model_calls"], 2)
        self.assertEqual(
            [item["role"] for item in diagnostics["images"]],
            ["overview", "app_grid"],
        )
        self.assertLessEqual(diagnostics["images"][0]["jpeg_bytes"], 28000)
        self.assertLessEqual(diagnostics["images"][1]["jpeg_bytes"], 24000)

    def test_unknown_overview_uses_app_specific_state_rois(self) -> None:
        provider = FakeProvider(
            [
                {"base_state": "unknown", "confidence": 0.90},
                {
                    "base_state": "douyin_video",
                    "confidence": 0.88,
                    "reason": "顶部是抖音推荐页",
                },
                {
                    "confidence": 0.91,
                    "heart_state": "unliked",
                    "targets": {"heart": [700, 450]},
                    "target_bounds": {"heart": [620, 390, 780, 510]},
                },
            ]
        )
        observer = DashScopePageObserver(provider)
        result = observer.observe(
            operation="douyin.batch_interact",
            params={"like": True, "target_count": 1},
            frames=[patterned_frame() for _ in range(4)],
            controller_context={},
        )
        self.assertEqual(result.state, "douyin_video")
        self.assertEqual(result.heart_state, "unliked")
        self.assertEqual(
            [item["role"] for item in observer.last_observation_diagnostics["images"]],
            ["overview", "douyin_page_evidence", "page_state_right"],
        )

    def test_false_live_overview_is_neutrally_corrected_by_rois(self) -> None:
        provider = FakeProvider(
            [
                {
                    "base_state": "douyin_live_preview",
                    "confidence": 0.99,
                    "live_evidence": ["直播中", "点击进入直播间"],
                    "live_evidence_bounds": {
                        "直播中": [80, 700, 240, 750],
                        "点击进入直播间": [300, 420, 700, 500],
                    },
                },
                {
                    "base_state": "douyin_video",
                    "confidence": 0.95,
                    "visible_texts": ["作者", "作品说明"],
                    "live_evidence": [],
                    "live_evidence_bounds": {},
                },
                {
                    "confidence": 0.96,
                    "heart_state": "unliked",
                    "targets": {"heart": [700, 450], "comments": [700, 590]},
                    "target_bounds": {
                        "heart": [620, 390, 780, 510],
                        "comments": [620, 530, 780, 650],
                    },
                    "live_evidence": [],
                    "live_evidence_bounds": {},
                },
            ]
        )
        observer = DashScopePageObserver(provider)
        result = observer.observe(
            operation="douyin.like_current",
            params={},
            frames=[patterned_frame() for _ in range(4)],
            controller_context={},
        )
        self.assertEqual(result.state, "douyin_video")
        self.assertEqual(result.heart_state, "unliked")
        self.assertEqual(len(provider.calls), 3)
        detail_text = provider.calls[1]["messages"][0]["content"][0]["text"]
        self.assertIn("暂定候选状态：unknown", detail_text)
        self.assertNotIn("已确认主页面状态", detail_text)

    def test_like_operation_always_rechecks_heart_in_high_res_roi(self) -> None:
        provider = FakeProvider(
            [
                {
                    "base_state": "douyin_video",
                    "confidence": 0.99,
                    "heart_state": "liked",
                    "targets": {"heart": [850, 500]},
                    "target_bounds": {"heart": [820, 470, 880, 530]},
                },
                {
                    "confidence": 0.98,
                    "heart_state": "unliked",
                    "targets": {
                        "heart": [650, 540],
                        "like": {"unexpected": "protocol alias"},
                    },
                    "target_bounds": {"heart": [570, 480, 730, 600]},
                },
            ]
        )
        observer = DashScopePageObserver(provider)
        result = observer.observe(
            operation="douyin.like_current",
            params={},
            frames=[patterned_frame() for _ in range(4)],
            controller_context={},
        )
        self.assertEqual(result.state, "douyin_video")
        self.assertEqual(result.heart_state, "unliked")
        self.assertNotIn("like", result.targets)
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(
            observer.last_observation_diagnostics["images"][1]["role"],
            "right_actions",
        )

    def test_unknown_blocking_overlay_only_requests_close_roi(self) -> None:
        provider = FakeProvider(
            [
                {
                    "base_state": "unknown",
                    "overlays": ["permission_dialog"],
                    "confidence": 0.92,
                },
                {
                    "confidence": 0.91,
                    "targets": {"close_overlay": [800, 120]},
                    "target_bounds": {"close_overlay": [740, 70, 860, 170]},
                },
            ]
        )
        observer = DashScopePageObserver(provider)
        result = observer.observe(
            operation="douyin.search",
            params={"keyword": "测试"},
            frames=[patterned_frame() for _ in range(4)],
            controller_context={},
        )
        self.assertIn("permission_dialog", result.blocking_overlays)
        self.assertIn("close_overlay", result.targets)
        self.assertEqual(
            [item["role"] for item in observer.last_observation_diagnostics["images"]],
            ["overview", "overlay_close"],
        )

    def test_unstable_frames_stop_before_model_call(self) -> None:
        provider = FakeProvider([])
        observer = DashScopePageObserver(provider)
        first = patterned_frame()
        second = Image.new("RGB", first.size, "white")
        with self.assertRaises(VisionAgentError):
            observer.observe(
                operation="douyin.search",
                params={"keyword": "测试"},
                frames=[first, second, first, second],
                controller_context={},
            )
        self.assertEqual(provider.calls, [])
        self.assertEqual(observer.last_observation_diagnostics["model_calls"], 0)


if __name__ == "__main__":
    unittest.main()
