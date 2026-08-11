import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import build_vision_review_queue as review_queue


def _item(item_id: str, image: str, action: str) -> dict:
    return {
        "id": item_id,
        "run": "test-run",
        "step": 1,
        "goal": "test",
        "outcome": "failed",
        "image": image,
        "review_status": "failure_candidate",
        "recorded_decision": {
            "action": action,
            "screen_type": "keyboard",
        },
    }


class VisionReviewQueueTests(unittest.TestCase):
    def test_identical_image_distance_is_zero(self) -> None:
        image = Image.new("RGB", (120, 200), "white")
        signature = review_queue.image_signature(image)
        self.assertEqual(
            review_queue.signature_distance(signature, signature),
            0.0,
        )

    def test_clearly_different_images_have_high_distance(self) -> None:
        black = review_queue.image_signature(
            Image.new("RGB", (120, 200), "black")
        )
        white = review_queue.image_signature(
            Image.new("RGB", (120, 200), "white")
        )
        self.assertGreater(review_queue.signature_distance(black, white), 200)

    def test_similar_images_cluster_together(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            replay_root = Path(temp_dir)
            images = replay_root / "history" / "images"
            images.mkdir(parents=True)
            first = images / "first.png"
            second = images / "second.png"
            Image.new("RGB", (120, 200), (100, 100, 100)).save(first)
            Image.new("RGB", (120, 200), (102, 102, 102)).save(second)

            items = [
                _item("first", "history/images/first.png", "tap"),
                _item("second", "history/images/second.png", "tap"),
            ]
            with patch.object(review_queue, "REPLAY_ROOT", replay_root):
                clusters = review_queue.cluster_items(items, threshold=4.0)

        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0]["members"]), 2)

    def test_conflicting_actions_require_review(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            replay_root = Path(temp_dir)
            history_dir = replay_root / "history"
            images = history_dir / "images"
            images.mkdir(parents=True)
            Image.new("RGB", (120, 200), "white").save(images / "same.png")
            history_index = history_dir / "history_index.json"
            history_index.write_text(
                json.dumps(
                    {
                        "items": [
                            _item("tap", "history/images/same.png", "tap"),
                            _item(
                                "clear",
                                "history/images/same.png",
                                "clear_text",
                            ),
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch.object(review_queue, "REPLAY_ROOT", replay_root),
                patch.object(review_queue, "HISTORY_INDEX", history_index),
                patch.object(
                    review_queue,
                    "QUEUE_JSON",
                    history_dir / "review_queue.json",
                ),
                patch.object(
                    review_queue,
                    "QUEUE_HTML",
                    history_dir / "review_queue.html",
                ),
            ):
                payload = review_queue.build_review_queue(threshold=4.0)

        self.assertEqual(payload["stats"]["clusters"], 1)
        self.assertEqual(payload["stats"]["conflict_clusters"], 1)
        self.assertTrue(payload["clusters"][0]["needs_conflict_review"])
        rendered = review_queue.render_review_html(payload)
        self.assertIn('src="images/same.png"', rendered)
        self.assertNotIn('src="history/images/same.png"', rendered)
        self.assertEqual(rendered.count('src="images/same.png"'), 3)


if __name__ == "__main__":
    unittest.main()
