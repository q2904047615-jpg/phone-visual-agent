"""Install the lucky-bag plugin's extra dependencies and the pinned upstream detector.

    python -m pva.plugins.lucky_bag.setup
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

from pva.config import ROOT

SETTINGS = Path(__file__).resolve().parent / "settings.json"


def main() -> None:
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    subprocess.run([sys.executable, "-m", "pip", "install", "rapidocr-onnxruntime==1.4.4",
                    "opencv-python-headless"], check=True)
    if settings["mode"] == "upstream":
        directory = ROOT / "external" / "douyin_guaji"
        directory.mkdir(parents=True, exist_ok=True)
        url = (f"https://raw.githubusercontent.com/pokemonzlj/douyin_guaji/"
               f"{settings['upstream_commit']}/douyin_fudai.py")
        data = urllib.request.urlopen(url, timeout=60).read()
        if hashlib.sha256(data).hexdigest() != settings["upstream_sha256"]:
            raise SystemExit("上游文件校验失败，未安装。")
        (directory / "douyin_fudai.py").write_bytes(data)
        (directory / "verified-source.json").write_text(json.dumps(
            {"commit": settings["upstream_commit"], "sha256": settings["upstream_sha256"]}), encoding="utf-8")
    print("福袋插件依赖已安装。")


if __name__ == "__main__":
    main()
