"""Nine-point touch calibration for a robot device.

1. Start the page server:   python -m pva.tools.calibration_page
2. Open http://<this PC's IP>:8770/ in the phone's browser, keep it in front of the camera.
3. Collect (fit):          python -m pva.tools.calibrate collect --device device-local-01
4. Reset the page (reload it), then validate:
                           python -m pva.tools.calibrate validate --device device-local-01

``collect`` taps the centre target (to enter full screen) and then the nine
magenta targets *without* correction, and fits the camera->arm transform.
``validate`` repeats the nine points *with* the new correction and enables it
only if every touch lands within tolerance. Press Esc to abort at any time.
Do not run this while the web console is running a task on the same controller.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from pva.actions import Action
from pva.config import load_config
from pva.devices import build_device
from pva.devices.calibration import (Affine2D, CalibrationError, build_calibration, build_coverage,
                                     save_calibration)
from pva.util import read_json


def request_json(url: str, method: str = "GET", payload: object | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, method=method, headers={"Content-Type": "application/json"} if data else {})
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def locate_magenta_target(frame: Image.Image) -> tuple[int, int]:
    pixels = np.asarray(frame.convert("RGB")).astype(np.int16)
    red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    mask = (red > 110) & (blue > 100) & (green < 140) & (red - green > 45) & (blue - green > 35)
    ys, xs = np.nonzero(mask)
    if len(xs) < 24:
        raise CalibrationError("画面中没有清晰的紫红色校准靶点。")
    cx, cy = int(np.median(xs)), int(np.median(ys))
    near = (np.abs(xs - cx) <= 40) & (np.abs(ys - cy) <= 40)
    xs, ys = xs[near], ys[near]
    if len(xs) < 24 or xs.max() - xs.min() > 90 or ys.max() - ys.min() > 90:
        raise CalibrationError("靶点像素异常（模糊或页面不对）。")
    return round(float(xs.mean())), round(float(ys.mean()))


def stable_target(device, timeout: float = 6.0) -> tuple[Image.Image, tuple[int, int]]:
    """Same target in two consecutive frames, so page transitions are ignored."""
    deadline, previous, error = time.monotonic() + timeout, None, ""
    while time.monotonic() < deadline:
        frame = device.capture().convert("RGB")
        try:
            center = locate_magenta_target(frame)
            if previous and math.dist(center, previous) <= 5:
                return frame, center
            previous = center
        except CalibrationError as exc:
            error, previous = str(exc), None
        time.sleep(0.25)
    raise CalibrationError(f"没有检测到稳定的靶点。{error}")


def wait_page(base: str, phases: set[str], sequence: int | None = None, timeout: float = 8.0) -> dict:
    deadline, latest = time.monotonic() + timeout, {}
    while time.monotonic() < deadline:
        latest = request_json(f"{base}/api/page-state")
        if latest.get("phase") in phases and (sequence is None or int(latest.get("sequence", -1)) == sequence):
            return latest
        time.sleep(0.2)
    raise CalibrationError(f"校准页没有进入 {sorted(phases)}：{latest}")


def wait_sample(base: str, count: int, timeout: float = 8.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        samples = request_json(f"{base}/api/samples").get("samples", [])
        if len(samples) > count:
            return samples[-1]
        time.sleep(0.2)
    raise CalibrationError("落笔后手机校准页没有收到触点（手机没被点到？）。")


def raw_tap(device, frame: Image.Image, pixel: tuple[int, int]) -> None:
    """Tap a camera pixel with the calibration correction bypassed."""
    grid = (round(pixel[0] * 1000 / (frame.width - 1)), round(pixel[1] * 1000 / (frame.height - 1)))
    saved, device.calibration.transform = device.calibration.transform, None
    try:
        device.execute(Action("tap", point=grid), frame.size)
    finally:
        device.calibration.transform = saved


def run(mode: str, device_id: str, base: str) -> None:
    config = load_config()
    entry = config.device(device_id)
    if entry.driver != "robot" or entry.calibration is None:
        raise SystemExit("只有配置了 calibration 路径的机械臂设备需要标定。")
    device = build_device(config, device_id)
    device.prepare()
    correction = None
    if mode == "validate":
        payload = read_json(entry.calibration, default=None)
        if not payload or not payload.get("accepted_fit"):
            raise SystemExit("还没有通过拟合的标定文件，请先运行 collect。")
        correction = Affine2D.from_json(payload["target_to_command"])

    state = wait_page(base, {"fullscreen_setup", "calibration"})
    if state["phase"] == "fullscreen_setup":
        frame, center = stable_target(device)
        print("点击中央靶点进入全屏…")
        raw_tap(device, frame, center)
        state = wait_page(base, {"calibration"}, sequence=0)
    if int(state.get("sequence", 0)) != 0:
        raise SystemExit("校准页不是从第 1 点开始，请刷新手机上的校准页后重试。")

    samples, frame_size = [], None
    for sequence in range(9):
        frame, target = stable_target(device)
        frame_size = frame_size or frame.size
        if frame.size != frame_size:
            raise CalibrationError("采集过程中相机画面尺寸变化。")
        command = target
        if correction is not None:
            cx, cy = correction.apply(target[0] / (frame.width - 1), target[1] / (frame.height - 1))
            command = (round(cx * (frame.width - 1)), round(cy * (frame.height - 1)))
        count = len(request_json(f"{base}/api/samples").get("samples", []))
        raw_tap(device, frame, command)
        record = wait_sample(base, count)
        width, height = float(record["viewport_width"]), float(record["viewport_height"])
        samples.append({
            "sequence": sequence, "desired_frame": list(target), "command_frame": list(command),
            "target_dom": [record["target_x"] / width, record["target_y"] / height],
            "actual_dom": [record["actual_x"] / width, record["actual_y"] / height],
            "viewport_size": [width, height],
        })
        error = math.hypot(record["actual_x"] - record["target_x"], record["actual_y"] - record["target_y"])
        print(f"第 {sequence + 1}/9 点：误差 {error:.1f} CSS px")
        wait_page(base, {"calibration", "complete"}, sequence=sequence + 1)

    if mode == "collect":
        payload = build_calibration(samples, frame_size)
        save_calibration(payload, entry.calibration)
        print(json.dumps(payload["fit"], ensure_ascii=False))
        if not payload["accepted_fit"]:
            raise SystemExit("拟合误差或覆盖范围不合格，标定未启用。请检查相机/手机位置后重来。")
        print(f"拟合完成：{entry.calibration}。刷新手机校准页后运行 validate 启用。")
        return

    distances = [math.hypot((s["actual_dom"][0] - s["target_dom"][0]) * s["viewport_size"][0],
                            (s["actual_dom"][1] - s["target_dom"][1]) * s["viewport_size"][1]) for s in samples]
    rms = math.sqrt(sum(d * d for d in distances) / len(distances))
    coverage = build_coverage([(s["desired_frame"][0] / (frame_size[0] - 1), s["desired_frame"][1] / (frame_size[1] - 1))
                               for s in samples])
    passed = rms <= 8.0 and max(distances) <= 16.0 and coverage["sufficient"]
    payload = read_json(entry.calibration)
    payload["validation"] = {"rms_error_css_px": round(rms, 3), "max_error_css_px": round(max(distances), 3),
                             "coverage": coverage, "coverage_passed": coverage["sufficient"], "passed": passed,
                             "samples": samples}
    payload["validated_at"] = datetime.now(timezone.utc).isoformat()
    payload["validated"] = payload["enabled"] = passed
    save_calibration(payload, entry.calibration)
    print(f"验证 RMS {rms:.1f}px，最大 {max(distances):.1f}px：{'通过，纠偏已启用' if passed else '未通过，纠偏保持关闭'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="机械臂九点触控标定")
    parser.add_argument("mode", choices=("collect", "validate"))
    parser.add_argument("--device", required=True)
    parser.add_argument("--page", default="http://127.0.0.1:8770", help="校准页服务地址")
    args = parser.parse_args()
    base = args.page.rstrip("/")
    request_json(f"{base}/api/health")
    input(f"将用机械臂在 {args.device} 上点击 10 次（按 Esc 可中断）。确认手机已打开校准页后按回车开始…")
    run(args.mode, args.device, base)


if __name__ == "__main__":
    main()
