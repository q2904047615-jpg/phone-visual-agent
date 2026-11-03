import json
from unittest.mock import patch

import pytest
from PIL import Image

from pva.actions import ActionError, describe, parse_action
from pva.config import DeviceConfig
from pva.devices.calibration import Affine2D, Calibration, CalibrationError
from pva.model.prompt import build_messages, Shot
from pva.model.qwen import ModelError, extract_json


# -- action vocabulary -------------------------------------------------------------
def test_parse_actions_and_aliases():
    assert parse_action({"type": "tap", "point": [500.4, 10]}).point == (500, 10)
    assert parse_action({"action": "click", "point": [1, 2]}).type == "tap"
    named = parse_action({"type": "tap", "point": [1, 2], "target": " 顶部搜索框 "})
    assert named.target == "顶部搜索框" and describe(named) == "点击「顶部搜索框」 [1, 2]"
    assert parse_action({"type": "text", "text": "hi"}).type == "type"
    scroll = parse_action({"type": "scroll", "direction": "up"})
    assert scroll.type == "swipe" and scroll.start[1] > scroll.end[1]
    assert parse_action({"type": "launch", "app": "抖音"}, apps=["抖音"]).app == "抖音"
    assert parse_action({"type": "wait", "seconds": 999}).seconds == 60
    finish = parse_action({"type": "finish", "success": "false", "message": "x"})
    assert finish.success is False and describe(finish).startswith("无法完成")


@pytest.mark.parametrize("raw,needle", [
    ({"type": "tap", "point": [1001, 5]}, "超出"),
    ({"type": "tap"}, "point"),
    ({"type": "swipe", "start": [1, 1], "end": [1, 1]}, "不能相同"),
    ({"type": "type", "text": ""}, "text"),
    ({"type": "scroll", "direction": "diagonal"}, "direction"),
    ({"type": "teleport"}, "未知"),
])
def test_parse_action_rejects(raw, needle):
    with pytest.raises(ActionError, match=needle):
        parse_action(raw)


def test_launch_must_be_registered_and_capability_checked():
    with pytest.raises(ActionError, match="不在本机直启列表"):
        parse_action({"type": "launch", "app": "微信"}, apps=["抖音"])
    with pytest.raises(ActionError, match="不支持"):
        parse_action({"type": "type", "text": "hi"}, supported={"tap"})


# -- model reply parsing -------------------------------------------------------------
def test_extract_json_tolerates_fences_and_prose():
    text = '好的。\n```json\n{"action": {"type": "back"}, "reply": "含有 } 括号"}\n```'
    assert extract_json(text)["reply"] == "含有 } 括号"
    with pytest.raises(ModelError):
        extract_json("没有 JSON")


def test_messages_carry_all_history_images_and_history_text():
    frame = Image.new("RGB", (540, 960))
    messages = build_messages(task="t", history=[{"step": 1, "action": "点击 [1, 2]", "reason": "r", "result": "已执行"}],
                              notes=["补充"], apps=["抖音"], actions=["tap"], past_shots=[Shot(frame, "a"), Shot(frame, "b")],
                              current=frame, thumbnail=(36, 64), current_max=(540, 960))
    content = messages[1]["content"]
    assert sum(part["type"] == "image_url" for part in content) == 3
    assert "补充" in content[0]["text"] and "第1步：点击 [1, 2]" in content[0]["text"]
    assert content[-2]["type"] == "image_url"  # CURRENT is last


# -- calibration -------------------------------------------------------------------------
def test_calibration_identity_when_disabled(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"enabled": False}), encoding="utf-8")
    assert Calibration(path).correct(100, 200, (540, 960)) == (100, 200)
    assert Calibration(None).correct(5, 6, (540, 960)) == (5, 6)


def test_calibration_applies_affine_and_rejects_aspect_change(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"version": 2, "enabled": True, "validated": True, "frame_size": [540, 960],
                                "coverage": {"sufficient": True},
                                "target_to_command": [[1, 0, 0.01], [0, 1, -0.02]]}), encoding="utf-8")
    calibration = Calibration(path)
    assert calibration.active
    assert calibration.correct(500, 500, (810, 1440)) == (510, 480)  # isotropic resize is fine
    with pytest.raises(CalibrationError):
        calibration.correct(500, 500, (540, 540))


def test_affine_inverse_roundtrip():
    transform = Affine2D(((1.1, 0.02, 0.01), (-0.01, 0.95, 0.03)))
    x, y = transform.inverse().apply(*transform.apply(0.3, 0.7))
    assert abs(x - 0.3) < 1e-9 and abs(y - 0.7) < 1e-9


# -- robot device coordinate mapping (Win32 calls patched) -----------------------------
def _seller_stubs(**overrides):
    """Patch every Win32-touching seller call a robot action makes; override any by name."""
    from contextlib import ExitStack

    defaults = {"find_window": (1, "智联新途"), "clear_seller_camera_overlay": None, "_check_escape": None,
                "seller_popup_text": None, "configure_single_click_count": None}
    stack = ExitStack()
    for name, value in {**defaults, **overrides}.items():
        target = f"pva.devices.seller_window.{name}"
        stack.enter_context(patch(target, side_effect=value) if callable(value) else patch(target, return_value=value))
    return stack


def test_robot_tap_maps_grid_to_client_pixels(tmp_path):
    from pva.config import DEFAULT_CONTROLLER
    from pva.devices.base import grid_to_pixel
    from pva.devices.robot import RobotDevice

    device = RobotDevice(DeviceConfig(id="r", name="r", driver="robot"), DEFAULT_CONTROLLER)
    calls = []
    with _seller_stubs(click_client_point=lambda hwnd, x, y, *a, **k: calls.append((x, y)) or {
            "input_events_dispatched": True}):
        receipt = device.tap((500, 1000), (540, 960))
        device.navigate("back", (540, 960))
    assert calls[0] == (270, 959)
    assert calls[1] == grid_to_pixel((685, 976), (540, 960))   # nav keys use the tap grid mapping
    assert receipt["channel"] == "robot"


def test_robot_double_tap_restores_click_count():
    from pva.config import DEFAULT_CONTROLLER
    from pva.devices.robot import RobotDevice

    device = RobotDevice(DeviceConfig(id="r", name="r", driver="robot"), DEFAULT_CONTROLLER)
    events = []
    with _seller_stubs(configure_click_count=lambda h, n: events.append(("count", n)),
                       configure_single_click_count=lambda h: events.append(("count", 1)),
                       click_client_point=lambda *a, **k: events.append("click") or {}):
        device.double_tap((10, 10), (540, 960))
    assert events == [("count", 2), "click", ("count", 1)]


def test_robot_single_clicks_recheck_click_count_first():
    from pva.config import DEFAULT_CONTROLLER
    from pva.devices.robot import RobotDevice

    device = RobotDevice(DeviceConfig(id="r", name="r", driver="robot"), DEFAULT_CONTROLLER)
    events = []
    with _seller_stubs(configure_single_click_count=lambda h: events.append("count"),
                       click_client_point=lambda *a, **k: events.append("click") or {}):
        device.tap((10, 10), (540, 960))
        device.navigate("home", (540, 960))
    assert events == ["count", "click", "count", "click"]


def test_robot_stops_on_controller_message_box():
    from pva.config import DEFAULT_CONTROLLER
    from pva.devices.base import DeviceError
    from pva.devices.robot import RobotDevice

    device = RobotDevice(DeviceConfig(id="r", name="r", driver="robot"), DEFAULT_CONTROLLER)
    popups = iter([None, "点击次数必须是一个整数"])   # none before the click, one right after it
    with _seller_stubs(seller_popup_text=lambda h: next(popups), click_client_point=lambda *a, **k: {}):
        with pytest.raises(DeviceError, match="点击后控制端弹出提示“点击次数必须是一个整数”"):
            device.tap((10, 10), (540, 960))
    clicks = []
    with _seller_stubs(seller_popup_text="点击次数必须是一个整数",
                       click_client_point=lambda *a, **k: clicks.append(1) or {}):
        with pytest.raises(DeviceError, match="点击前"):
            device.tap((10, 10), (540, 960))
    assert clicks == []   # an open message box blocks the next action before it is sent


def test_robot_nav_keys_use_touch_calibration():
    from types import SimpleNamespace

    from pva.config import DEFAULT_CONTROLLER
    from pva.devices.base import grid_to_pixel
    from pva.devices.robot import RobotDevice

    device = RobotDevice(DeviceConfig(id="r", name="r", driver="robot"), DEFAULT_CONTROLLER)
    device.calibration = SimpleNamespace(correct=lambda x, y, frame_size: (x + 20, y - 30))
    calls = []
    with _seller_stubs(click_client_point=lambda hwnd, x, y, *a, **k: calls.append((x, y)) or {}):
        device.navigate("back", (540, 960))
        device.tap((685, 976), (540, 960))
    # The back key is corrected exactly like a tap on the same camera grid point.
    assert calls[0] == calls[1] == grid_to_pixel((705, 946), (540, 960))


# -- camera frame sanity check ------------------------------------------------------
def _frame(screen=None, *, size=(540, 960), overlays=True):
    from PIL import ImageDraw
    image = Image.new("RGB", size, "black")
    draw = ImageDraw.Draw(image)
    w, h = size
    if screen:
        x0, y0, x1, y1 = screen
        draw.rectangle((x0 * w, y0 * h, x1 * w, y1 * h), fill=(200, 200, 210))
    if overlays:   # vendor coordinate text at the top, machine-position buttons at the bottom
        draw.rectangle((0, 0, w * 0.55, h * 0.03), fill="white")
        draw.rectangle((0, h * 0.985, w, h), fill="white")
    return image


def test_frame_check_reports_black_frame():
    from pva.devices.frame_check import check_frame
    assert "全黑" in check_frame(_frame(None), None)


def test_frame_check_accepts_a_lit_screen_without_expected_region():
    from pva.devices.frame_check import check_frame, screen_region
    image = _frame((0.07, 0.0, 0.93, 0.91))
    assert check_frame(image, None) is None
    x0, y0, x1, y1 = screen_region(image)
    assert abs(x0 - 0.07) < 0.02 and abs(x1 - 0.93) < 0.02 and y0 < 0.06 and abs(y1 - 0.91) < 0.02


def test_frame_check_compares_with_expected_region():
    from pva.devices.frame_check import check_frame
    expected = (0.07, 0.0, 0.93, 0.91)
    assert check_frame(_frame((0.08, 0.0, 0.92, 0.90)), expected) is None
    message = check_frame(_frame((0.01, 0.0, 0.87, 0.74)), expected)   # what the camera showed this morning
    assert message and "位置" in message and "0.74" in message


def test_device_config_parses_screen_region(tmp_path, monkeypatch):
    from pva.config import ConfigError, load_config
    monkeypatch.delenv("PVA_MOCK", raising=False)
    path = tmp_path / "devices.json"
    path.write_text(json.dumps({"devices": [{"id": "r", "driver": "robot", "screen_region": [0.07, 0, 0.93, 0.91]}]}),
                    encoding="utf-8")
    assert load_config(path).device("r").screen_region == (0.07, 0.0, 0.93, 0.91)
    path.write_text(json.dumps({"devices": [{"id": "r", "driver": "robot", "screen_region": [0.9, 0, 0.1, 1]}]}),
                    encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


# -- cross-process device lease (shared with the legacy console and the platform connector) --
def test_lease_path_matches_legacy_naming(tmp_path):
    from pva.devices.lease import lease_path
    assert lease_path(tmp_path, "device-local-01").name == "device_1d4c34d50bc62ea902fdc2e4.lease"


def _robot_manager(tmp_path, monkeypatch):
    from pva.config import load_config
    from pva.devices import DeviceManager
    monkeypatch.delenv("PVA_MOCK", raising=False)
    path = tmp_path / "devices.json"
    path.write_text(json.dumps({"devices": [{"id": "device-local-01", "driver": "robot"},
                                            {"id": "sim", "driver": "mock"}]}), encoding="utf-8")
    return DeviceManager(load_config(path), lease_dir=tmp_path / "leases")


def test_reserve_refuses_a_device_held_by_another_program(tmp_path, monkeypatch):
    from pva.devices import DeviceBusy
    from pva.devices.lease import InterProcessLease, lease_path
    devices = _robot_manager(tmp_path, monkeypatch)
    other = InterProcessLease(lease_path(tmp_path / "leases", "device-local-01"), owner_id="connector",
                              metadata={"device_id": "device-local-01", "session_id": "connector-windows-local-01"})
    assert other.acquire()
    try:
        with pytest.raises(DeviceBusy, match="connector-windows-local-01"):
            devices.reserve("device-local-01", "task-1")
        assert devices.owner("device-local-01") is None
    finally:
        other.release()
    devices.reserve("device-local-01", "task-1")
    devices.release("device-local-01", "task-1")


def test_reserve_holds_the_lease_until_release(tmp_path, monkeypatch):
    from pva.devices.lease import InterProcessLease, lease_path
    devices = _robot_manager(tmp_path, monkeypatch)
    path = lease_path(tmp_path / "leases", "device-local-01")
    devices.reserve("device-local-01", "task-1")
    devices.reserve("device-local-01", "task-1")          # resume by the same owner is fine
    assert InterProcessLease.active_payload(path)["session_id"] == "task-1"
    assert not InterProcessLease(path, owner_id="legacy", metadata={}).acquire()
    devices.release("device-local-01", "task-1")
    assert InterProcessLease.active_payload(path) is None
    devices.reserve("sim", "task-2")                     # the simulated phone needs no lease
    assert not lease_path(tmp_path / "leases", "sim").exists()


def test_frame_check_does_not_flag_dark_pages():
    from PIL import ImageDraw
    from pva.devices.frame_check import check_frame
    expected = (0.07, 0.0, 0.93, 0.91)
    # A lit phone showing a black page ("直播已结束"): only a little white text and one icon.
    image = _frame(None)
    draw = ImageDraw.Draw(image)
    draw.rectangle((200, 60, 340, 80), fill="white")
    draw.ellipse((240, 110, 300, 170), outline="white", width=4)
    assert check_frame(image, None) is None
    assert check_frame(image, expected) is None
    # Even on a dark page, light outside where the screen should be means the camera moved.
    draw.rectangle((0, 300, 20, 700), fill="white")
    assert "位置" in check_frame(image, expected)


def test_frame_check_does_not_flag_a_bright_bottom_sheet():
    from pva.devices.frame_check import check_frame
    from PIL import ImageDraw
    # Dark live room with a white panel over its lower half: the panel is page content, not a moved screen.
    image = _frame(None)
    ImageDraw.Draw(image).rectangle((0.07 * 540, 0.46 * 960, 0.93 * 540, 0.95 * 960), fill=(220, 220, 220))
    assert check_frame(image, (0.07, 0.0, 0.93, 0.91)) is None


def test_mark_action_draws_on_a_copy_only():
    from pva.model.prompt import mark_action

    image = Image.new("RGB", (540, 960), "white")
    marked, note = mark_action(image, {"type": "tap", "point": [500, 500]})
    radius = round(540 * 0.03)
    assert marked.getpixel((round(499 * 0.539) + radius, round(959 * 0.5)))[1] < 50   # red ring edge
    assert image.getpixel((270, 480)) == (255, 255, 255) and "红圈" in note
    swiped, note = mark_action(image, {"type": "swipe", "start": [500, 800], "end": [500, 200]})
    assert swiped.getpixel((270, 480))[1] < 50 and "红线" in note
    assert mark_action(image, {"type": "home"}) == (image, "")
    assert mark_action(image, None) == (image, "")
