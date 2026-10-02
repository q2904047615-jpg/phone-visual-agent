"""Three additional saved-image targets; reuse the frozen minimal probe, no hardware."""
import argparse
from datetime import datetime
import hashlib
from pathlib import Path
import sys

POC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC))
from experiments import probe_minimal_wire_grounding as base

OUT = POC / "output/other_buttons_minimal_20260908"
ORDER = ("emoji", "more", "home_icon")
TARGETS = {
    "emoji": "输入栏中位于发送按钮左侧的圆形笑脸表情按钮",
    "more": "聊天页面右上角的三个点更多按钮",
    "home_icon": "手机底部系统导航栏正中间的圆角方形Home图标",
}
# Visible reference regions in the exact source JPEG, never part of model input.
REGIONS = {"emoji": [512, 1093, 555, 1139], "more": [603, 7, 635, 25],
           "home_icon": [349, 1233, 380, 1263]}


def bind():
    base.OUT = OUT
    base.ORDER = ORDER
    base.TARGETS = TARGETS
    base.REGIONS = REGIONS


def wrapper_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def prepare():
    if OUT.exists():
        raise RuntimeError("New campaign already exists; do not reset")
    base.selftest()  # Existing helper's original tests, before binding new targets.
    bind()
    bodies = [base.build(target)[0] for target in ORDER]
    urls = [body["messages"][0]["content"][0]["image_url"]["url"] for body in bodies]
    assert len(set(urls)) == 1
    for target, point in zip(ORDER, ([740, 872], [860, 12], [506, 977])):
        assert base.score(point, target)["inside_visible_region"]
        assert not base.score([0, 500], target)["inside_visible_region"]
    OUT.mkdir()
    for index, body in enumerate(bodies, 1):
        base.save(f"{index}_request.json", body)
    base.save("preflight.json", {
        "created_at": datetime.now().astimezone().isoformat(), "source": str(base.SOURCE),
        "source_sha256": hashlib.sha256(base.SOURCE.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
        "wrapper_sha256": wrapper_hash(), "jpeg_sha256": base.build(ORDER[0])[1],
        "production_hashes": base.hashes(), "request_hashes": [base.digest(b) for b in bodies],
        "order": ORDER, "regions": REGIONS, "max_calls": 3, "max_attempts": 1, "phone_actions": 0,
        "authorization": "User requested checking other buttons, continued saved-image-only diagnosis; no physical clicks.",
        "stop": "3 attempts, any transport error, or two consecutive contract errors; no retries.",
        "limitation": "Each new target once; not repeated accuracy, new screen, live navigation or mechanical acceptance."})
    print("PREFLIGHT_PASS: identical source JPEG, three target scores, max_calls=3, remote_calls=0")


def recognize(index):
    bind()
    if index not in range(1, 4):
        raise RuntimeError("Only three calls authorized")
    pre = base.read(OUT / "preflight.json")
    assert pre["wrapper_sha256"] == wrapper_hash()
    base.recognize(index)
    if index == 3:
        base.save("stopped.json", {"reserved_calls": 3, "resume_allowed": False, "phone_actions": 0})


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
