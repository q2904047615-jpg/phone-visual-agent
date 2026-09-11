"""Approved isolated full-frame ROI -> enlarged-crop point screening; no devices.

This deliberately is NOT the production observation protocol or a tool-loop replica.
Only model outputs choose ROI and point. Human regions are offline scoring data.
"""
import argparse
import base64
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

from PIL import Image

POC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC))
from experiments import probe_minimal_wire_grounding as base

OUT = POC / "output/zoom_grounding_20260908"
ORDER = ("voice", "more", "send", "emoji")
TARGETS = {**base.TARGETS,
    "more": "聊天页面右上角的三个点更多按钮",
    "emoji": "输入栏中位于发送按钮左侧的圆形笑脸表情按钮"}
REGIONS = {**base.REGIONS, "more": [603, 7, 635, 25], "emoji": [512, 1093, 555, 1139]}
ROI_PROMPT = ("在这张{w}×{h}像素的完整图片中找到{target}。"
    "选择一个适合放大观察的矩形区域，完整包含该目标及少量周围上下文。不要给点击点。"
    "以整张图片左上角为原点，横纵轴均归一化到0..999。"
    "只返回JSON数组[x1,y1,x2,y2]；无法唯一辨认目标则返回null。")
POINT_PROMPT = ("这张{w}×{h}像素图片是原画面中一个区域的放大图。"
    "原目标为：{target}。请在这张放大图中定位该目标，给出其内部一个可点击点。"
    "只使用当前放大图坐标，以左上角为原点，横纵轴均归一化到0..999；不要换算回原图。"
    "只返回JSON数组[x,y]；目标不在图内、不完整或不能唯一辨认则返回null。")


def save(name, value):
    base.atomic_replace_bytes(OUT / name, base.json_bytes(value))


def source_image():
    body, sha = base.build("send")
    url = body["messages"][0]["content"][0]["image_url"]["url"]
    image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
    return url, image, sha


def request(url, prompt):
    return {"model": "qwen3.7-plus", "temperature": 0.0, "enable_thinking": True,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": prompt}]}]}


def parse_roi(raw):
    text = raw.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    box = json.loads(text)
    if box is None:
        return None
    if not isinstance(box, list) or len(box) != 4 or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        or not 0 <= v <= 999 for v in box
    ) or not (box[0] < box[2] and box[1] < box[3]):
        raise ValueError("Expected one finite ordered normalized ROI")
    return box


def make_crop(image, roi):
    w, h = image.size
    # Inclusive normalized pixel centers -> exclusive Pillow crop edge.
    box = [math.floor(roi[0] * (w-1) / 999), math.floor(roi[1] * (h-1) / 999),
           math.ceil(roi[2] * (w-1) / 999) + 1, math.ceil(roi[3] * (h-1) / 999) + 1]
    crop = image.crop(box)
    enlarged = crop.resize((crop.width * 2, crop.height * 2), Image.Resampling.LANCZOS)
    stream = io.BytesIO()
    enlarged.save(stream, format="PNG")
    data = stream.getvalue()
    return data, {"crop_box_exclusive": box, "crop_size": list(crop.size),
                  "zoom_size": list(enlarged.size), "scale": 2,
                  "png_sha256": hashlib.sha256(data).hexdigest()}


def to_original(point, geometry):
    left, top, _, _ = geometry["crop_box_exclusive"]
    zw, zh = geometry["zoom_size"]
    scale = geometry["scale"]
    # Inverse of the resize's pixel-center transformation, no semantic correction.
    return [left + (point[0] * (zw-1) / 999 + .5) / scale - .5,
            top + (point[1] * (zh-1) / 999 + .5) / scale - .5]


def score(xy, target):
    x1, y1, x2, y2 = REGIONS[target]
    inside = xy is not None and x1 <= xy[0] <= x2 and y1 <= xy[1] <= y2
    result = {"inside_visible_region": inside}
    if xy is not None and target in ("voice", "emoji"):
        result["inside_reference_ellipse_diagnostic"] = (
            ((xy[0]-(x1+x2)/2)/((x2-x1)/2))**2 +
            ((xy[1]-(y1+y2)/2)/((y2-y1)/2))**2 <= 1
        )
    return result


def frozen_hashes():
    return {str(p.relative_to(POC)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(base.__file__), base.SOURCE)}


def stop(reason):
    save("stopped.json", {"reason": reason, "reserved_calls": len(list(OUT.glob("*_attempt.json"))),
        "max_calls": 8, "resume_allowed": False, "phone_actions": 0})


def guard(index, pre):
    if (OUT / "stopped.json").exists() or index not in range(1, 9):
        raise RuntimeError("Campaign stopped or budget exhausted")
    if index != len(list(OUT.glob("*_attempt.json"))) + 1:
        raise RuntimeError("No repetition, concurrency or out-of-order calls")
    if index > 1 and not (OUT / f"{index-1}_result.json").exists():
        raise RuntimeError("Previous call unresolved; do not retry")
    if frozen_hashes() != pre["frozen_hashes"] or base.hashes() != pre["production_hashes"]:
        raise RuntimeError("Frozen input, experiment or production drift")


def prepare():
    if OUT.exists():
        raise RuntimeError("Campaign already exists; never reset")
    url, im, sha = source_image()
    assert sha == "453cd9b3754b9669867f615a4bff90dc4b9988979199046a53a3dc0bb5f7d324"
    OUT.mkdir()
    roi_bodies = [request(url, ROI_PROMPT.format(w=im.width, h=im.height, target=TARGETS[t])) for t in ORDER]
    for index, body in zip((1, 3, 5, 7), roi_bodies):
        save(f"{index}_request.json", body)
    save("preflight.json", {"created_at": datetime.now().astimezone().isoformat(),
        "order": ORDER, "max_calls": 8, "max_attempts": 1, "phone_actions": 0,
        "authorization": "User approved up to 8 saved-image Qwen calls; no production integration or phone actions.",
        "jpeg_sha256": sha, "frozen_hashes": frozen_hashes(), "production_hashes": base.hashes(),
        "roi_request_hashes": [base.digest(b) for b in roi_bodies], "regions_for_scoring_only": REGIONS,
        "roi_prompt": ROI_PROMPT, "point_prompt": POINT_PROMPT, "zoom": "fixed 2x Lanczos, lossless PNG",
        "stop": "Any transport/contract/null failure, final point outside original visible reference, drift, or eight calls. Never retry.",
        "limitations": "Four target pairs, one saved frame. Not single-factor attribution, tool-loop replica, accuracy rate or production/live validation.",
        "reference": "https://github.com/QwenLM/Qwen-Agent/blob/main/examples/cookbook_think_with_images.ipynb"})
    print("PREPARED: max_calls=8; source JPEG frozen; model selects ROI; no remote/device IO")


def recognize(index):
    pre = base.read(OUT / "preflight.json")
    guard(index, pre)
    target = ORDER[(index-1)//2]
    stage = "roi" if index % 2 else "point"
    body = base.read(OUT / f"{index}_request.json")
    if stage == "roi":
        assert base.digest(body) == pre["roi_request_hashes"][(index-1)//2]
    else:
        prior = base.read(OUT / f"{index-1}_result.json")
        assert base.digest(body) == prior["next_request_sha256"]
    provider = base.transport.DashScopeVisionProvider(enable_thinking=True, max_attempts=1)
    assert provider.configured and provider.model == body["model"]
    assert provider.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert provider.model_config.request_options() == {"enable_thinking": True}
    with (OUT / f"{index}_attempt.json").open("x", encoding="utf-8") as handle:
        json.dump({"started_at": datetime.now().astimezone().isoformat(), "stage": stage, "target": target}, handle)
    original_post = base.transport.httpx.post
    calls = 0

    def captured_post(url, **kwargs):
        nonlocal calls
        assert calls == 0 and url == provider.base_url + "/chat/completions" and kwargs["json"] == body
        calls += 1
        response = original_post(url, **kwargs)
        base.atomic_replace_bytes(OUT / f"{index}_response_body.json", response.content)
        save(f"{index}_http.json", {"status_code": response.status_code, "calls": calls})
        return response

    result = {"index": index, "stage": stage, "target": target, "phone_actions": 0, "ok": False}
    reason = None
    try:
        with patch.object(base.transport.httpx, "post", captured_post):
            raw = provider._chat(body["messages"], max_tokens=None, timeout=60, max_attempts=1)
        save(f"{index}_content.json", {"raw": raw})
        parsed = parse_roi(raw) if stage == "roi" else base.parse(raw)
        result["parsed"] = parsed
        if parsed is None:
            reason = "model_reports_no_unique_target"
        elif stage == "roi":
            _, im, _ = source_image()
            data, geometry = make_crop(im, parsed)
            base.atomic_replace_bytes(OUT / f"{target}_zoom.png", data)
            crop_url = "data:image/png;base64," + base64.b64encode(data).decode("ascii")
            w, h = geometry["zoom_size"]
            next_body = request(crop_url, POINT_PROMPT.format(w=w, h=h, target=TARGETS[target]))
            save(f"{index+1}_request.json", next_body)
            result.update(ok=True, geometry=geometry, next_request_sha256=base.digest(next_body))
        else:
            geometry = base.read(OUT / f"{index-1}_result.json")["geometry"]
            xy = to_original(parsed, geometry)
            result.update(ok=True, geometry=geometry, original_image_point=xy, **score(xy, target))
            if not result["inside_visible_region"]:
                reason = "candidate_miss_no_improvement_or_regression"
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        reason = "transport_or_contract_or_experiment_error"
    finally:
        result.update(network_attempts=provider.last_network_attempts, usage=provider.last_usage,
            response_id=provider.last_request_id, response_model=provider.last_response_model,
            finish_reason=provider.last_finish_reason, production_unchanged=base.hashes() == pre["production_hashes"])
        save(f"{index}_result.json", result)
        if reason or index == 8 or not result["production_unchanged"]:
            stop(reason or ("completed_screening_budget" if index == 8 else "production_drift"))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "recognize"))
    parser.add_argument("--index", type=int)
    parser.add_argument("--allow-remote", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.allow_remote:
        recognize(args.index)
    else:
        parser.error("Explicit --allow-remote required")
