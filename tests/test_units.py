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
def test_robot_tap_maps_grid_to_client_pixels(tmp_path):
    from pva.config import DEFAULT_CONTROLLER
    from pva.devices.robot import RobotDevice

    device = RobotDevice(DeviceConfig(id="r", name="r", driver="robot"), DEFAULT_CONTROLLER)
    calls = []
    with patch("pva.devices.seller_window.find_window", return_value=(1, "智联新途")), \
         patch("pva.devices.seller_window.click_client_point",
               side_effect=lambda hwnd, x, y, *a, **k: calls.append((x, y)) or {"input_events_dispatched": True}), \
         patch("pva.devices.seller_window.clear_seller_camera_overlay"), \
         patch("pva.devices.seller_window._check_escape"):
        receipt = device.tap((500, 1000), (540, 960))
        device.navigate("back", (540, 960))
    assert calls[0] == (270, 959)
    assert calls[1] == (round(540 * 0.685), round(960 * 0.976))
    assert receipt["channel"] == "robot"


def test_robot_double_tap_restores_click_count():
    from pva.config import DEFAULT_CONTROLLER
    from pva.devices.robot import RobotDevice

    device = RobotDevice(DeviceConfig(id="r", name="r", driver="robot"), DEFAULT_CONTROLLER)
    events = []
    with patch("pva.devices.seller_window.find_window", return_value=(1, "t")), \
         patch("pva.devices.seller_window.configure_click_count", side_effect=lambda h, n: events.append(("count", n))), \
         patch("pva.devices.seller_window.configure_single_click_count", side_effect=lambda h: events.append(("count", 1))), \
         patch("pva.devices.seller_window.click_client_point", side_effect=lambda *a, **k: events.append("click") or {}), \
         patch("pva.devices.seller_window.clear_seller_camera_overlay"), \
         patch("pva.devices.seller_window._check_escape"):
        device.double_tap((10, 10), (540, 960))
    assert events == [("count", 2), "click", ("count", 1)]
