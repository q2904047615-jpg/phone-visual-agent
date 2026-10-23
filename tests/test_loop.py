import time

import pytest

from conftest import ScriptedModel, step
from pva.agent.manager import TaskManager
from pva.devices import DeviceBusy, DeviceManager


def wait_for(session, statuses, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if session.status in statuses:
            return session.status
        time.sleep(0.02)
    raise AssertionError(f"status stayed {session.status}: {session.detail} {session.result_message}")


def manager(mock_config, tmp_path, replies):
    devices = DeviceManager(mock_config)
    model = ScriptedModel(replies)
    return TaskManager(mock_config, devices, model, tmp_path / "out"), devices, model


NOTE_FLOW = [
    step({"type": "launch", "app": "备忘录"}),
    step({"type": "tap", "point": [844, 860]}, last="success"),
    step({"type": "tap", "point": [501, 339]}, last="success"),
    step({"type": "type", "text": "买牛奶"}, last="success"),
    step({"type": "tap", "point": [909, 76]}, last="success"),
    step({"type": "finish", "success": True, "message": "已保存笔记"}, last="success"),
]


def test_full_task_on_simulated_phone(mock_config, tmp_path):
    tasks, devices, model = manager(mock_config, tmp_path, NOTE_FLOW)
    session = tasks.start(device_id="sim", task="新建一条笔记：买牛奶")
    assert wait_for(session, {"succeeded", "failed"}) == "succeeded"
    phone = devices.get("sim").phone
    assert phone.state.notes == ["买牛奶"]
    assert session.actions_used == 5 and session.observations_used == 6
    assert session.result_message == "已保存笔记"
    # Every request carries the full history: request n has n-1 HISTORY images plus CURRENT.
    last = model.requests[-1][1]["content"]
    images = [part for part in last if part["type"] == "image_url"]
    assert len(images) == 6
    text = last[0]["text"]
    assert "第4步：输入「买牛奶」" in text and "执行结果：已执行" in text
    # The device is free again.
    assert devices.owner("sim") is None
    assert (session.run_dir / "session.json").is_file()
    assert (session.run_dir / "step_006.jpg").is_file()


def test_one_task_per_controller(mock_config, tmp_path):
    tasks, devices, _ = manager(mock_config, tmp_path, [step({"type": "wait", "seconds": 5})] * 3)
    first = tasks.start(device_id="sim", task="等一下")
    with pytest.raises(DeviceBusy):
        tasks.start(device_id="sim", task="另一个任务")
    second = tasks.start(device_id="sim2", task="另一台手机")
    first.cancel()
    second.cancel()
    assert first.status == "cancelled" and second.status == "cancelled"


def test_payment_requires_confirmation(mock_config, tmp_path):
    replies = [step({"type": "tap", "point": [500, 500]}, risk="payment"),
               step({"type": "finish", "message": "done"})]
    tasks, devices, _ = manager(mock_config, tmp_path, replies)
    session = tasks.start(device_id="sim", task="付款")
    assert wait_for(session, {"needs_confirmation"}) == "needs_confirmation"
    assert session.actions_used == 0 and session.pending is not None
    assert devices.owner("sim") == session.id
    session.approve()
    assert wait_for(session, {"succeeded"}) == "succeeded"
    assert session.actions_used == 1


def test_step_mode_and_reject(mock_config, tmp_path):
    replies = [step({"type": "home"}), step({"type": "back"}), step({"type": "finish"})]
    tasks, devices, _ = manager(mock_config, tmp_path, replies)
    session = tasks.start(device_id="sim", task="回桌面", mode="step")
    wait_for(session, {"awaiting_approval"})
    session.reject("不要按 home")
    wait_for(session, {"awaiting_approval"})
    assert session.pending.type == "back"
    session.approve()
    wait_for(session, {"succeeded"})
    assert session.actions_used == 1
    assert session.notes[0]["text"] == "不要按 home"


def test_bad_model_output_retries_then_fails(mock_config, tmp_path):
    replies = ["not json", step({"type": "teleport"}), {"action": {"type": "tap", "point": [5000, 1]}}]
    tasks, _, _ = manager(mock_config, tmp_path, replies)
    session = tasks.start(device_id="sim", task="x")
    assert wait_for(session, {"failed"}) == "failed"
    assert "连续 3 次模型错误" in session.result_message
    assert session.actions_used == 0


def test_budget_pauses_and_resume(mock_config, tmp_path):
    replies = [step({"type": "back"})] * 2 + [step({"type": "finish"})]
    tasks, _, _ = manager(mock_config, tmp_path, replies)
    session = tasks.start(device_id="sim", task="x", max_actions=2)
    assert wait_for(session, {"paused"}) == "paused"
    assert session.actions_used == 2 and "动作次数已达上限" in session.detail
    session.resume(max_actions=5)
    assert wait_for(session, {"succeeded"}) == "succeeded"


def test_model_can_report_impossible_task(mock_config, tmp_path):
    tasks, _, _ = manager(mock_config, tmp_path, [step({"type": "finish", "success": False, "message": "没有该联系人"})])
    session = tasks.start(device_id="sim", task="给赵六发消息")
    assert wait_for(session, {"failed"}) == "failed"
    assert session.result_message == "没有该联系人"


def test_recents_goes_home_first(mock_config, tmp_path):
    replies = [step({"type": "launch", "app": "设置"}), step({"type": "recents"}), step({"type": "finish"})]
    tasks, devices, model = manager(mock_config, tmp_path, replies)
    session = tasks.start(device_id="sim", task="打开最近任务")
    wait_for(session, {"succeeded"})
    assert devices.get("sim").phone.state.screen == "recents"
    assert session.steps[1].extra_screenshots == ["step_002_home.jpg"]
    labels = [part["text"] for part in model.requests[-1][1]["content"] if part["type"] == "text"]
    assert any("先回到桌面" in label for label in labels)
