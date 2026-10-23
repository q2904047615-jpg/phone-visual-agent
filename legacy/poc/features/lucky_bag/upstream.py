"""Run only the pinned upstream detector; never its unattended main loop.

Upstream files are fetched locally by setup_trial.py, not redistributed.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
from PIL import Image


class UpstreamDetector:
    def __init__(self, directory: Path, offset=0):
        source = directory / "douyin_fudai.py"
        manifest = json.loads((directory / "verified-source.json").read_text(encoding="utf-8"))
        if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["sha256"]:
            raise RuntimeError("上游检测源码哈希不匹配，请重新运行安装脚本。")
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        class_node = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == "fudai_analyse")
        method = next(n for n in class_node.body if isinstance(n,ast.FunctionDef) and n.name == "check_have_fudai")
        namespace = {"os":os, "time":time, "Image":Image, "__file__":str(source)}
        exec(compile(ast.Module(body=[method], type_ignores=[]),str(source),"exec"),namespace)
        self.function = namespace["check_have_fudai"]
        self.directory, self.offset = directory, offset
        self.directory.joinpath("pic").mkdir(exist_ok=True)

    def locate(self, image):
        # Each upstream scan consumes this single saved screenshot. No hidden
        # captures, captcha actions, global stdout replacement or room switching.
        image.save(self.directory / "pic/screenshot.png")
        owner = SimpleNamespace(device_id="trial", y_pianyi=self.offset,
            resolution_ratio_x=image.width, resolution_ratio_y=image.height,
            device_offsets={"trial":self.offset}, last_find_fudai_time=0)
        owner.operation = SimpleNamespace(delay=lambda _:None, get_screenshot=lambda _:None,
            update_config_with_offsets=lambda *args:None)
        owner.check_zhibo_is_closed = lambda: False
        owner.deal_robot_analyse = lambda: None
        x = self.function(owner)
        if x is False:
            return None
        return round((x+25)*image.width/1080), round((430+owner.y_pianyi)*image.height/2400)
