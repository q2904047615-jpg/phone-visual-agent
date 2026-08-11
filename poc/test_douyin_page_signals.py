from __future__ import annotations

import unittest
from pathlib import Path

from PIL import Image

from douyin_page_signals import detect_douyin_local_signals


ROOT = Path(__file__).resolve().parent


class DouyinPageSignalRegressionTests(unittest.TestCase):
    def test_current_real_live_preview_is_detected(self) -> None:
        sample = ROOT / "output" / "actual_live_page_20260810.jpg"
        self.assertTrue(sample.exists(), f"缺少实机直播样本：{sample}")
        signals = detect_douyin_local_signals(Image.open(sample).convert("RGB"))
        self.assertIsNotNone(signals.live_preview_badge)
        self.assertIsNone(signals.live_room_close)

    def test_current_real_ordinary_video_is_not_live(self) -> None:
        sample = ROOT / "output" / "supervised_after_observe.jpg"
        self.assertTrue(sample.exists(), f"缺少实机普通视频样本：{sample}")
        signals = detect_douyin_local_signals(Image.open(sample).convert("RGB"))
        self.assertIsNone(signals.live_preview_badge)
        self.assertIsNone(signals.live_room_close)


if __name__ == "__main__":
    unittest.main()
