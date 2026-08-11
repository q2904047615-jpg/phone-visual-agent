from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

from PIL import Image, ImageDraw

import robot_gui_poc as legacy
from robot_core import RobotController


def capture_click_burst(
    *,
    relative_x: int,
    relative_y: int,
    output_dir: Path,
    before_seconds: float = 0.6,
    after_seconds: float = 1.2,
    interval_seconds: float = 0.04,
) -> dict[str, object]:
    """Record the seller preview while one harmless physical tap executes.

    This is a diagnostic, not a workflow action.  It preserves the complete
    camera burst so XY motion, pen descent and phone response can be inspected
    separately instead of guessing another coordinate offset.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    robot = RobotController()
    hwnd, title = legacy.find_window(robot.title)
    frames: list[tuple[float, Image.Image]] = []
    capture_errors: list[str] = []
    started = time.monotonic()
    stop_at = started + before_seconds + after_seconds + 0.8

    def recorder() -> None:
        while time.monotonic() < stop_at:
            try:
                frame = legacy.camera_crop(
                    legacy.capture_client(hwnd), legacy.DEFAULT_CAMERA_HEIGHT
                ).convert("RGB")
                frames.append((time.monotonic() - started, frame))
            except Exception as exc:  # Preserve a partial burst for diagnosis.
                capture_errors.append(str(exc))
            time.sleep(interval_seconds)

    thread = threading.Thread(target=recorder, name="click-burst-recorder", daemon=True)
    thread.start()
    time.sleep(before_seconds)
    tap_started = time.monotonic() - started
    pixel = robot.vision_tap_relative(relative_x, relative_y)
    tap_finished = time.monotonic() - started
    time.sleep(after_seconds)
    thread.join(timeout=2.0)

    for index, (elapsed, frame) in enumerate(frames):
        frame.save(output_dir / f"frame_{index:03d}_{elapsed:06.3f}.jpg", quality=92)

    # Keep a compact visual artifact that can be inspected without playing a
    # video.  Prefer frames around the actual down/up interval.
    selected = [
        item
        for item in frames
        if tap_started - 0.30 <= item[0] <= tap_finished + 0.55
    ]
    if len(selected) > 12:
        step = (len(selected) - 1) / 11
        selected = [selected[round(i * step)] for i in range(12)]
    if selected:
        thumb_width = 270
        thumb_height = round(selected[0][1].height * thumb_width / selected[0][1].width)
        columns = 4
        rows = (len(selected) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + 28)), "black")
        draw = ImageDraw.Draw(sheet)
        for index, (elapsed, frame) in enumerate(selected):
            thumb = frame.resize((thumb_width, thumb_height))
            left = (index % columns) * thumb_width
            top = (index // columns) * (thumb_height + 28)
            sheet.paste(thumb, (left, top + 28))
            phase = "TAP" if tap_started <= elapsed <= tap_finished else ""
            draw.text((left + 5, top + 6), f"{elapsed:.3f}s {phase}", fill="white")
        sheet.save(output_dir / "contact_sheet.jpg", quality=94)

    result: dict[str, object] = {
        "window_title": title,
        "relative_coordinate": [relative_x, relative_y],
        "commanded_pixel": list(pixel),
        "tap_started_seconds": tap_started,
        "tap_finished_seconds": tap_finished,
        "captured_frames": len(frames),
        "capture_errors": capture_errors,
        "contact_sheet": str(output_dir / "contact_sheet.jpg"),
    }
    (output_dir / "diagnostic.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Record one physical-tap camera burst.")
    parser.add_argument("--x", type=int, required=True, help="1000-grid X coordinate")
    parser.add_argument("--y", type=int, required=True, help="1000-grid Y coordinate")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = capture_click_burst(relative_x=args.x, relative_y=args.y, output_dir=args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
