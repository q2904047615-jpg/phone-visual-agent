"""Offline-only explicit image-pixel zoom candidate; not a production protocol.

No model, device, API or file-write entry point. A new authorized experiment must
construct new requests. Never reinterpret prior normalized responses as pixels.
Qwen3.7 documents normalized coordinates; pixel adherence is UNVERIFIED.
"""
import base64
import io
import json
import math

from PIL import Image


def _size(width, height):
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 2 for v in (width, height)):
        raise ValueError("Expected actual image dimensions >= 2")


def prompt(target, width, height, *, region):
    _size(width, height)
    task = ("选择一个完整包含目标及少量周围上下文的矩形区域，供后续放大观察。不要给点击点。"
            if region else "在当前放大图中给出目标内部一个可点击点，不要换算回原图。")
    result = ('{"coordinate_space":"image_pixels","bbox":[x1,y1,x2,y2]}'
              if region else '{"coordinate_space":"image_pixels","point":[x,y]}')
    return (f"当前图片真实尺寸为{width}×{height}像素。目标：{target}。{task}"
            f"仅使用当前图片的原始像素坐标：左上角为(0,0)，x范围0到{width-1}，y范围0到{height-1}。"
            "无需归一化或比例换算。只返回如下JSON结构：" + result +
            "。区域右下角为包含在区域内的像素位置。目标缺失、不完整或不能唯一辨认则返回null。")


def build_request(image_url, target, *, region):
    # Derive dimensions from the exact encoded image, never model-provided values.
    if not image_url.startswith(("data:image/jpeg;base64,", "data:image/png;base64,")):
        raise ValueError("Only explicit in-memory PNG/JPEG accepted")
    data = base64.b64decode(image_url.split(",", 1)[1], validate=True)
    with Image.open(io.BytesIO(data)) as image:
        w, h = image.size
    return {"model": "qwen3-vl-plus", "temperature": 0.0, "enable_thinking": True,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": prompt(target, w, h, region=region)}]}]}


def parse(raw, width, height, *, region):
    _size(width, height)
    text = raw.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    value = json.loads(text)
    if value is None:
        return None
    field = "bbox" if region else "point"
    if (not isinstance(value, dict) or set(value) != {"coordinate_space", field}
            or value["coordinate_space"] != "image_pixels"):
        raise ValueError("Expected explicit image_pixels contract; no legacy inference")
    coordinates = value[field]
    count = 4 if region else 2
    if (not isinstance(coordinates, list) or len(coordinates) != count or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            or v < 0 or v > (width-1 if i % 2 == 0 else height-1)
            for i, v in enumerate(coordinates))):
        raise ValueError("Coordinates outside actual image dimensions")
    if region and not (coordinates[0] < coordinates[2] and coordinates[1] < coordinates[3]):
        raise ValueError("ROI must have positive area")
    return coordinates


def crop_and_enlarge(image, roi):
    # Validate against decoded image before Pillow, which otherwise permits padding.
    parse(json.dumps({"coordinate_space": "image_pixels", "bbox": roi}), *image.size, region=True)
    bounds = [math.floor(roi[0]), math.floor(roi[1]), math.ceil(roi[2])+1, math.ceil(roi[3])+1]
    crop = image.crop(bounds)
    enlarged = crop.resize((crop.width*2, crop.height*2), Image.Resampling.LANCZOS)
    return enlarged, {"bounds_exclusive": bounds, "scale": 2, "zoom_size": list(enlarged.size)}


def to_original(raw, geometry):
    point = parse(raw, *geometry["zoom_size"], region=False)
    if point is None:
        return None
    left, top, _, _ = geometry["bounds_exclusive"]
    scale = geometry["scale"]
    return [left + (point[0]+.5)/scale-.5, top + (point[1]+.5)/scale-.5]
