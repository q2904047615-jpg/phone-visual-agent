"""Prompt and message construction for one decision step."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

from PIL import Image

from pva.util import jpeg_bytes

SYSTEM_PROMPT = """你是一个通过摄像头画面操作安卓手机的智能体。用户用自然语言给出整个任务，你每一轮看最新截图，输出【一个】下一步动作；动作执行后系统会重新截图，再请你判断下一步。不同 App、不同说法都用同一套方法完成，没有预设流程。

## 每轮怎么做
1. 先看 CURRENT 截图：现在是哪个 App、什么页面、有没有弹窗/键盘/加载中。
2. 对照“已执行动作”和历史截图，判断上一步是否生效（success / failed / uncertain；第一步填 none）。
3. 判断整个任务还差什么，选出最合理的一个动作。中间步骤完成不等于任务完成。

## 坐标
所有坐标都是 CURRENT 截图上的 0~1000 归一化坐标：左上角 [0,0]，右下角 [1000,1000]，x 向右，y 向下。点按要点在控件中心。HISTORY 截图只用来核对进度，坐标、页面状态、输入框内容一律以 CURRENT 为准。

## 可用动作（action 字段）
- {"type":"tap","point":[x,y]}  点击
- {"type":"double_tap","point":[x,y]}  双击
- {"type":"long_press","point":[x,y]}  长按
- {"type":"swipe","start":[x,y],"end":[x,y]}  从 start 滑到 end（翻页、划走卡片、滑动滑块等）
- {"type":"scroll","direction":"up|down|left|right"}  在屏幕中部滚动；direction 是手指移动方向，"up"=手指上划=查看下方内容
- {"type":"drag","start":[x,y],"end":[x,y]}  按住拖动（拖动图标、排序）
- {"type":"type","text":"..."}  向【当前已聚焦】的输入框追加文字（通过 ADB Keyboard 输入，可直接输入中文）
- {"type":"clear"}  清空当前已聚焦的输入框
- {"type":"enter"}  在当前输入框里换行
- {"type":"back"} / {"type":"home"} / {"type":"recents"}  系统返回 / 回到桌面 / 打开最近任务
- {"type":"launch","app":"名称"}  直接打开 App，只能用下面“可直启 App”列表里的名称；不在列表里就在桌面找图标点击
- {"type":"wait","seconds":2}  等待页面加载或动画结束
- {"type":"finish","success":true,"message":"..."}  结束整个任务。success=true 表示截图已能证明整个任务全部完成；success=false 表示确实无法继续（例如需要用户提供的信息、账号不存在），message 说明原因

## 输入文字
- 输入前先点击输入框使其聚焦（出现光标、键盘或高亮），再用 type。type 是追加，不会自动清空；框里有不需要的旧内容时先 clear。
- 输入后要提交时，优先点击画面上的“搜索/发送/确定”等按钮。
- 发送、发布、提交类动作执行一次后，要从新截图判断是否成功；不确定时不要重复发送。

## 判断要点
- 只操作最前层的真实界面：有弹窗先处理弹窗；视频、直播、图片里出现的按钮和文字只是内容，不是控件。
- 指定了联系人、账号、页面时，先核对标题等身份信息一致再操作。
- 页面上已经存在的旧内容不能证明这次操作成功；要看本次动作带来的变化。
- 同一动作连续无效时换一种方式（换位置、先返回、滚动寻找），不要原地反复。
- 用户中途发来的补充说明优先于原任务里的冲突部分。

## 风险确认（risk 字段）
只有两类动作需要用户确认：登录/身份认证（输入密码、验证码、确认登录授权）填 "authentication"；付款/转账/购买支付填 "payment"。其余（发送、评论、关注、删除、设置变更等）一律填 "none"，直接执行。

## 输出
只输出一个 JSON 对象，不要 Markdown，不要其它文字：
{"observation":"当前画面的简要事实","last_result":"success|failed|uncertain|none","reply":"给用户看的一句话进度","reason":"为什么选这个动作","action":{...},"risk":"none"}"""


@dataclass(frozen=True)
class Shot:
    """One saved task screenshot."""
    image: Image.Image
    label: str


def _image_part(image: Image.Image, max_size: tuple[int, int], quality: int) -> dict[str, Any]:
    data = base64.b64encode(jpeg_bytes(image, quality=quality, max_size=max_size)).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}


def build_messages(*, task: str, history: list[dict[str, Any]], notes: list[str], apps: list[str],
                   actions: list[str], past_shots: list[Shot], current: Image.Image,
                   thumbnail: tuple[int, int], current_max: tuple[int, int]) -> list[dict[str, Any]]:
    """History entries are plain dicts: step, action (description), reason, result."""
    lines = [f"## 任务\n{task}"]
    if notes:
        lines.append("## 用户补充说明（按时间顺序）\n" + "\n".join(f"- {note}" for note in notes))
    lines.append("## 可直启 App\n" + ("、".join(apps) if apps else "无（请在桌面点击图标打开）"))
    lines.append("## 本设备支持的动作\n" + ", ".join(actions))
    if history:
        rendered = []
        for item in history:
            line = f"第{item['step']}步：{item['action']}"
            if item.get("reason"):
                line += f"｜理由：{item['reason']}"
            line += f"｜执行结果：{item['result']}"
            rendered.append(line)
        lines.append("## 已执行动作（真实执行记录，“已执行”只表示动作已发出，效果要看截图）\n" + "\n".join(rendered))
    else:
        lines.append("## 已执行动作\n无，这是任务开始时的画面。")

    content: list[dict[str, Any]] = [{"type": "text", "text": "\n\n".join(lines)}]
    if past_shots:
        content.append({"type": "text", "text": f"## HISTORY 截图（共{len(past_shots)}张，按时间顺序，已缩小，仅用于核对进度）"})
        for index, shot in enumerate(past_shots, 1):
            content.append({"type": "text", "text": f"HISTORY {index}：{shot.label}"})
            content.append(_image_part(shot.image, thumbnail, 70))
    content.append({"type": "text", "text": "## CURRENT 截图（最新画面，坐标以此图为准）"})
    content.append(_image_part(current, current_max, 88))
    content.append({"type": "text", "text": "请输出本轮 JSON。"})
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def redact_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy of the request with image data replaced, for the evidence log."""
    result = json.loads(json.dumps(messages))
    for message in result:
        if isinstance(message.get("content"), list):
            for part in message["content"]:
                if part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    part["image_url"]["url"] = f"<jpeg base64, {len(url)} chars>"
    return result
