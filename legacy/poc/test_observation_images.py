from __future__ import annotations
import unittest
from PIL import Image, ImageDraw
from agent.infrastructure.observation_images import (
    consensus_top_edge_obstructions,
    detect_top_edge_opaque_bands,
    measure_local_stability,
    measure_material_visual_transition,
)


def patterned_frame() -> Image.Image:
    image = Image.new("RGB", (540, 960), "#eee6db")
    draw = ImageDraw.Draw(image)
    for y in range(0, 960, 48):
        draw.rectangle((0, y, 540, min(y + 23, 959)), fill=(y % 255, 70, 130))
    for x in range(20, 520, 72):
        draw.ellipse((x, 220, x + 45, 265), fill="#13a87a")
    return image



class ObservationImageTests(unittest.TestCase):
    def test_detects_scaled_partial_width_top_obstruction_without_fixed_pixels(self) -> None:
        for size in ((540, 960), (810, 1440), (675, 1200)):
            image = Image.new("RGB", size, "#dedede")
            draw = ImageDraw.Draw(image)
            width, height = size
            band_right = round(width * 0.56)
            band_bottom = round(height * 0.045)
            draw.rectangle((0, 0, band_right, band_bottom), fill="black")
            draw.rectangle(
                (
                    round(width * 0.025),
                    round(height * 0.012),
                    round(width * 0.32),
                    round(height * 0.022),
                ),
                fill="white",
            )

            detected = detect_top_edge_opaque_bands(image)

            self.assertEqual(1, len(detected))
            self.assertEqual("top_edge_opaque_band", detected[0].kind)
            self.assertLessEqual(abs(detected[0].bounds[2] - 560), 35)
            self.assertLessEqual(abs(detected[0].bounds[3] - 45), 20)

    def test_top_obstruction_detector_ignores_status_bars_and_letterboxing(self) -> None:
        full_status = Image.new("RGB", (540, 960), "#dddddd")
        ImageDraw.Draw(full_status).rectangle((0, 0, 539, 45), fill="black")
        letterboxed = Image.new("RGB", (540, 960), "#dddddd")
        ImageDraw.Draw(letterboxed).rectangle((0, 0, 35, 959), fill="black")

        self.assertEqual((), detect_top_edge_opaque_bands(full_status))
        self.assertEqual((), detect_top_edge_opaque_bands(letterboxed))

    def test_top_obstruction_requires_stable_frame_consensus(self) -> None:
        clear = Image.new("RGB", (540, 960), "#dddddd")
        obstructed = clear.copy()
        ImageDraw.Draw(obstructed).rectangle((0, 0, 300, 44), fill="black")

        self.assertEqual(
            (),
            consensus_top_edge_obstructions(
                [obstructed, clear.copy(), clear.copy()]
            ),
        )
        self.assertEqual(
            1,
            len(
                consensus_top_edge_obstructions(
                    [clear.copy(), obstructed, obstructed.copy()]
                )
            ),
        )



    def test_local_stability_rejects_oscillating_recent_frames(self) -> None:
        frame = patterned_frame()
        stable = measure_local_stability([frame.copy() for _ in range(4)])
        self.assertTrue(stable.stable)
        self.assertEqual(stable.frame_count, 4)

        changed = Image.new("RGB", frame.size, "white")
        unstable = measure_local_stability([frame, changed, frame, changed])
        self.assertFalse(unstable.stable)

    def test_local_stability_accepts_one_stale_leading_frame_after_three_converge(self) -> None:
        settled = patterned_frame()
        stale = Image.new("RGB", settled.size, "white")

        stability = measure_local_stability(
            [stale, settled.copy(), settled.copy(), settled.copy()],
            allow_leading_outlier=True,
        )

        self.assertTrue(stability.stable)
        self.assertIn("末尾3帧", stability.reason)

    def test_local_stability_keeps_strict_full_window_default_for_actions(self) -> None:
        settled = patterned_frame()
        stale = Image.new("RGB", settled.size, "white")

        stability = measure_local_stability(
            [stale, settled.copy(), settled.copy(), settled.copy()]
        )

        self.assertFalse(stability.stable)
        self.assertIn("完整采样窗口", stability.reason)

    def test_local_stability_rejects_change_with_only_two_new_frames(self) -> None:
        old = patterned_frame()
        new = Image.new("RGB", old.size, "white")

        stability = measure_local_stability(
            [old.copy(), old.copy(), new, new.copy()],
            allow_leading_outlier=True,
        )

        self.assertFalse(stability.stable)
        self.assertIn("末尾3帧", stability.reason)

    def test_material_transition_rejects_camera_noise_without_new_ui_state(self) -> None:
        before = patterned_frame()
        after = Image.new("RGB", before.size)
        after.paste(before)
        after = after.point(lambda value: min(255, value + 1))

        credential = measure_material_visual_transition(
            [before.copy() for _ in range(4)],
            [after.copy() for _ in range(4)],
        )

        self.assertFalse(credential["material"])
        self.assertLess(credential["max_tile_median_delta"], credential["minimum_tile_delta"])

    def test_material_transition_accepts_small_local_ui_change(self) -> None:
        before = patterned_frame()
        after = before.copy()
        ImageDraw.Draw(after).rectangle((420, 700, 500, 780), fill="#19b45b")

        credential = measure_material_visual_transition(
            [before.copy() for _ in range(4)],
            [after.copy() for _ in range(4)],
        )

        self.assertTrue(credential["material"])
        self.assertGreaterEqual(credential["max_tile_median_delta"], credential["minimum_tile_delta"])

if __name__ == "__main__":
    unittest.main()
