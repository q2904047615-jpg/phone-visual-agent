"""Second model call that turns the decision model's target description into tap coordinates.

The decision model (good at reading the screen) says what to touch, e.g. "右下角黄色圆形+按钮";
the locator model (good at precise coordinates) finds it on the same CURRENT screenshot.
Real-device comparison on 2026-11-04: in a full task context qwen3.5/3.6/3.7-plus drifted
4-8% towards the bottom-right on edge controls, while qwen3-vl-plus stayed within ~1%.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from pva.actions import ActionError, _point
from pva.model.prompt import _image_part
from pva.model.qwen import ModelError, VisionModel, extract_json

LOCATE_PROMPT = ("你在手机截图上定位一个控件。坐标是截图上的 0~1000 归一化坐标：左上角 [0,0]，右下角 [1000,1000]，"
                 "x 向右，y 向下。只输出 JSON：{\"point\":[x,y]}，point 是目标控件的中心。"
                 "画面上找不到这个控件时输出 {\"point\":null}。")


@dataclass(frozen=True)
class Located:
    point: tuple[int, int]
    elapsed_seconds: float
    tokens: int


def locate(model: VisionModel, image: Image.Image, target: str, *, max_size: tuple[int, int]) -> Located:
    """Raise ModelError when the locator cannot give a usable point."""
    messages = [{"role": "system", "content": LOCATE_PROMPT},
                {"role": "user", "content": [_image_part(image, max_size, 88), {"type": "text", "text": f"目标：{target}"}]}]
    reply = model.complete(messages)
    data = extract_json(reply.content)
    if data.get("point") is None:
        raise ModelError(f"定位模型在画面上找不到「{target}」。")
    try:
        point = _point(data.get("point"), "point")
    except ActionError as exc:
        raise ModelError(f"定位模型输出无效：{exc}") from exc
    return Located(point, reply.elapsed_seconds, int(reply.usage.get("total_tokens", 0)))
