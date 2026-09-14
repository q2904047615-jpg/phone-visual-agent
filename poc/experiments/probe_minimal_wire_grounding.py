"""Four approved saved-JPEG point probes, isolated from production and devices."""
import argparse
import base64
import hashlib
import io
import json
import math
from datetime import datetime
from pathlib import Path
import sys
from unittest.mock import patch

from PIL import Image

POC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC))
from agent.infrastructure import dashscope_vision_provider as transport
from agent.infrastructure.atomic_files import atomic_replace_bytes, json_bytes

OUT = POC / "output/minimal_wire_grounding_20260908"
SOURCE = POC / "output/web/generic_supervised_20260908_020120_f46c94ba/29fd9152aac14d1785f0d3af50483f83_model_request.json"
ORDER = ("send", "voice", "send", "voice")
TARGETS = {"send": "输入栏右侧标有‘发送’的绿色按钮", "voice": "输入栏左侧的圆形语音切换图标"}
# Human-reviewed visible bounds in the actual request JPEG; scoring only, never sent.
REGIONS = {"send": [570, 1092, 663, 1142], "voice": [74, 1093, 116, 1139]}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(name, value):
    atomic_replace_bytes(OUT / name, json_bytes(value))


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def hashes():
    paths = list((POC / "agent").rglob("*.py")) + list((POC / "agent").rglob("*.txt"))
    paths += list((POC / "static").rglob("*.js")) + list((POC / "static").rglob("*.html"))
    paths += [POC / "web_app.py", POC / "tap_calibration.json"]
    return {str(p.relative_to(POC)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def build(target):
    original = read(SOURCE)
    images = [c["image_url"]["url"] for c in original["messages"][1]["content"] if c["type"] == "image_url"]
    url = images[-1]
    data = base64.b64decode(url.split(",", 1)[1], validate=True)
    with Image.open(io.BytesIO(data)) as im:
        assert im.size == (720, 1280) and im.format == "JPEG"
    prompt = (f"定位图中{TARGETS[target]}，给出目标内部一个可点击点。"
              "坐标以整张图片左上角为原点，横纵轴均归一化到0..999。"
              "只返回JSON数组[x,y]；目标不可辨认则返回null。")
    body = {"model": "qwen3.7-plus", "temperature": 0.0, "enable_thinking": True,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": prompt}]}]}
    return body, hashlib.sha256(data).hexdigest()


def parse(raw):
    value = raw.strip()
    if value.startswith("```json") and value.endswith("```"):
        value = value[7:-3].strip()
    elif value.startswith("```") and value.endswith("```"):
        value = value[3:-3].strip()
    point = json.loads(value)
    if point is None:
        return None
    if not isinstance(point, list) or len(point) != 2 or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        or not 0 <= v <= 999 for v in point
    ):
        raise ValueError("Expected one finite normalized XY point")
    return point


def score(point, target):
    if point is None:
        return {"image_point": None, "inside_visible_region": False}
    xy = [point[0] * 719 / 999, point[1] * 1279 / 999]
    x1, y1, x2, y2 = REGIONS[target]
    return {"image_point": xy, "inside_visible_region": x1 <= xy[0] <= x2 and y1 <= xy[1] <= y2}


def selftest():
    a, image_hash = build("send")
    b, other_hash = build("voice")
    assert image_hash == other_hash == "453cd9b3754b9669867f615a4bff90dc4b9988979199046a53a3dc0bb5f7d324"
    assert a == build("send")[0]
    assert len(a["messages"]) == 1 and "response_format" not in a
    assert a["messages"][0]["content"][0] == b["messages"][0]["content"][0]
    assert a["enable_thinking"] is True
    assert parse("[850,870]") == [850, 870]
    assert parse("```json\n[850,870]\n```") == [850, 870]
    assert parse("null") is None
    for raw in ("[true,1]", "[1000,1]", "[1]", "[1,NaN]", '{"point":[1,2]}'):
        try:
            parse(raw)
        except ValueError:
            pass
        else:
            raise AssertionError(raw)
    assert score([850, 870], "send")["inside_visible_region"]
    assert not score([850, 914], "send")["inside_visible_region"]
    assert score([130, 870], "voice")["inside_visible_region"]
    print("SELFTEST_PASS: JPEG identity, request isolation, parsing, coordinate scaling and two target scores")


def prepare():
    if OUT.exists():
        raise RuntimeError("Campaign already exists; never reset")
    selftest()
    OUT.mkdir()
    bodies = [build(target)[0] for target in ORDER]
    for index, body in enumerate(bodies, 1):
        save(f"{index}_request.json", body)
    save("preflight.json", {"created_at": datetime.now().astimezone().isoformat(),
        "source": str(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "jpeg_sha256": build("send")[1], "request_hashes": [digest(b) for b in bodies],
        "production_hashes": hashes(), "order": ORDER, "regions": REGIONS,
        "max_calls": 4, "max_attempts": 1, "phone_actions": 0,
        "scope": "Same exact last JPEG; two targets, two identical requests each. No device/API IO.",
        "stop": "Four attempts, any transport error, or two consecutive contract errors. Valid misses do not stop planned repetitions.",
        "limitation": "Multi-setting minimal baseline, not single-factor causal proof or live acceptance."})
    print("PREPARED: 4 calls; remote_calls=0; phone_actions=0")


def recognize(index):
    pre = read(OUT / "preflight.json")
    if (OUT / "stopped.json").exists() or index != len(list(OUT.glob("*_attempt.json"))) + 1 or index not in range(1, 5):
        raise RuntimeError("Stopped, repeated, or out of order")
    if index > 1 and not (OUT / f"{index-1}_result.json").exists():
        raise RuntimeError("Previous attempt unresolved")
    assert hashes() == pre["production_hashes"]
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == pre["source_sha256"]
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == pre["script_sha256"]
    body = read(OUT / f"{index}_request.json")
    assert digest(body) == pre["request_hashes"][index-1]
    provider = transport.DashScopeVisionProvider(enable_thinking=True, max_attempts=1)
    assert provider.configured and provider.model == body["model"]
    assert provider.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert provider.model_config.request_options() == {"enable_thinking": True}
    with (OUT / f"{index}_attempt.json").open("x", encoding="utf-8") as handle:
        json.dump({"started_at": datetime.now().astimezone().isoformat(), "max_attempts": 1}, handle)
    original_post = transport.httpx.post
    calls = 0

    def captured_post(url, **kwargs):
        nonlocal calls
        assert calls == 0 and url == provider.base_url + "/chat/completions" and kwargs["json"] == body
        calls += 1
        response = original_post(url, **kwargs)
        atomic_replace_bytes(OUT / f"{index}_response_body.json", response.content)
        save(f"{index}_http.json", {"status_code": response.status_code, "calls": calls})
        return response

    result = {"index": index, "target": ORDER[index-1], "phone_actions": 0, "transport_ok": False, "parse_ok": False}
    try:
        with patch.object(transport.httpx, "post", captured_post):
            raw = provider._chat(body["messages"], max_tokens=None, timeout=60, max_attempts=1)
        result["transport_ok"] = True
        save(f"{index}_content.json", {"raw": raw})
        point = parse(raw)
        result.update(parse_ok=True, raw_point=point, **score(point, ORDER[index-1]))
    except Exception as exc:
        result["error_type"] = type(exc).__name__
    finally:
        result.update(network_attempts=provider.last_network_attempts, usage=provider.last_usage,
                      response_id=provider.last_request_id, response_model=provider.last_response_model,
                      finish_reason=provider.last_finish_reason, production_unchanged=hashes() == pre["production_hashes"])
        save(f"{index}_result.json", result)
        previous_bad = index > 1 and not read(OUT / f"{index-1}_result.json")["parse_ok"]
        if index == 4 or not result["transport_ok"] or (not result["parse_ok"] and previous_bad):
            save("stopped.json", {"reserved_calls": index, "resume_allowed": False, "phone_actions": 0})
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("selftest", "prepare", "recognize"))
    parser.add_argument("--index", type=int)
    parser.add_argument("--allow-remote", action="store_true")
    args = parser.parse_args()
    if args.mode == "selftest":
        selftest()
    elif args.mode == "prepare":
        prepare()
    elif args.allow_remote:
        recognize(args.index)
    else:
        parser.error("Explicit --allow-remote required")
