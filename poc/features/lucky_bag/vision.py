"""Local OCR and template recognition, isolated to the lucky-bag feature."""
from dataclasses import dataclass
from pathlib import Path
import re
import cv2
import numpy as np
from .flow import Page


@dataclass
class Token:
    text: str
    box: tuple[float, float, float, float]
    score: float = 1

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return round((x1+x2)/2), round((y1+y2)/2)


class LocalOcr:
    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self.engine = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1,
            det_limit_type="max", det_limit_side_len=1280)

    def read(self, image):
        result, _ = self.engine(np.asarray(image.convert("RGB"))[:, :, ::-1].copy())
        tokens = []
        for points, text, score in result or []:
            xs, ys = zip(*points)
            tokens.append(Token(text, (min(xs), min(ys), max(xs), max(ys)), float(score)))
        return tokens


class TemplateDetector:
    def __init__(self, directory: Path, threshold=.86):
        self.directory, self.threshold = directory, threshold
        self.last_score = 0.0

    def locate(self, image):
        source = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        image_scale = min(1, 720 / image.width)
        if image_scale < 1:
            source = cv2.resize(source, (round(image.width*image_scale),round(image.height*image_scale)))
        best = None
        self.last_score = 0.0
        # Use actual screen dimensions rather than inherited phone coordinates.
        for path in sorted(self.directory.glob("*.png")):
            reference = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if reference is None or float(reference.std()) < 3:
                continue
            for scale in (.45, .6, .75, .9, 1, 1.15, 1.35, 1.6, 1.9, 2.2):
                width, height = max(4, round(reference.shape[1]*scale*image_scale)), max(4, round(reference.shape[0]*scale*image_scale))
                if width >= source.shape[1] or height >= source.shape[0]:
                    continue
                template = cv2.resize(reference, (width,height))
                _, score, _, position = cv2.minMaxLoc(cv2.matchTemplate(source, template, cv2.TM_CCOEFF_NORMED))
                if score > self.last_score:
                    self.last_score = float(score)
                    best = (round((position[0]+width/2)/image_scale),round((position[1]+height/2)/image_scale))
        return best if self.last_score >= self.threshold else None


def interpret(tokens, size, bag=None):
    width, height = size
    page = Page(bag=bag)
    texts = [re.sub(r"\s", "", item.text) for item in tokens]
    page.summary = " / ".join(item.text for item in tokens)
    page.detail = any(t in {"福袋", "超级福袋", "参与条件"} for t in texts)
    page.joined = any(t in {"已参与", "参与成功", "参与成功等待开奖", "参与成功等待抽奖"}
        or "已成功参与福袋" in t for t in texts)
    page.lost = any("没抽中福袋" in t or "没有抽中" in t or t == "未中奖" for t in texts)
    page.closed = any("直播已结束" in t or "直播结束" in t or "主播已下播" in t for t in texts)
    page.live = bag is not None or any(t == "直播间" or "直播中" in t for t in texts)
    countdowns = []
    for token, text in zip(tokens, texts):
        if text in {"去发表评论", "一键发表评论", "发表评论"}:
            page.comment_button = token.center
        if text == "发送":
            page.send_button = token.center
            sx1, sy1, sx2, sy2 = token.box
            candidates = [t for t in tokens if t.box[2] < sx1 and abs(t.center[1]-token.center[1]) <= max(12, (sy2-sy1)*1.2)
                and t.box[0] >= width*.08 and not re.sub(r"[\s.。…]", "", t.text).startswith("说点什么")
                and t.text.strip() not in {"弹", "弹幕", "发送"}]
            page.prefilled = any(re.search(r"[\w\u4e00-\u9fff]", t.text) for t in candidates)
        if text in {"知道了", "我知道了"}:
            page.dismiss_button = token.center
        for match in re.finditer(r"(?<!\d)(\d{1,2})[:：](\d{2})(?!\d)", text):
            minute, second = map(int, match.groups())
            near_bag = bag is not None and abs(token.center[0]-bag[0]) < width*.12 and abs(token.center[1]-bag[1]) < height*.06
            near_label = "倒计时" in text or any("倒计时" in t.text and abs(t.center[1]-token.center[1]) < height*.08 and abs(t.center[0]-token.center[0]) < width*.3 for t in tokens)
            if second < 60 and token.center[1] > height*.10 and (near_bag or near_label):
                countdowns.append(minute*60+second)
    if countdowns:
        page.countdown = min(countdowns)
    return page
