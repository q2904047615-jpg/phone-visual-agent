import time

from fastapi.testclient import TestClient

from conftest import ScriptedModel, step
from pva.web.server import create_app


def client(mock_config, tmp_path, monkeypatch, replies):
    monkeypatch.setenv("PVA_OUTPUT", str(tmp_path / "out"))
    model = ScriptedModel(replies)
    app = create_app(mock_config, model)
    return TestClient(app), app, model


def wait_status(c, task_id, statuses, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = c.get(f"/api/tasks/{task_id}").json()
        if data["status"] in statuses:
            return data
        time.sleep(0.05)
    raise AssertionError(data)


def test_status_preview_and_page(mock_config, tmp_path, monkeypatch):
    c, _, _ = client(mock_config, tmp_path, monkeypatch, [])
    status = c.get("/api/status").json()
    assert [d["id"] for d in status["devices"]] == ["sim", "sim2"]
    assert status["plugins"]["qishui"]["ok"] and status["plugins"]["lucky_bag"]["ok"]
    assert c.get("/").status_code == 200
    image = c.get("/api/devices/sim/preview.jpg")
    assert image.status_code == 200 and image.content[:2] == b"\xff\xd8"


def test_task_lifecycle_over_http(mock_config, tmp_path, monkeypatch):
    replies = [step({"type": "launch", "app": "设置"}, reply="打开设置"), step({"type": "finish", "message": "好了"})]
    c, app, _ = client(mock_config, tmp_path, monkeypatch, replies)
    created = c.post("/api/tasks", json={"device_id": "sim", "task": "打开设置"}).json()
    done = wait_status(c, created["id"], {"succeeded"})
    assert done["result"] == "好了" and len(done["steps"]) == 2
    assert done["steps"][0]["action_text"] == "打开 App「设置」"
    shot = c.get(f"/api/tasks/{created['id']}/files/{done['steps'][0]['screenshot']}")
    assert shot.status_code == 200
    assert c.get(f"/api/tasks/{created['id']}/files/..%2Fsession.json").status_code == 404
    history = c.get("/api/tasks").json()["tasks"]
    assert history[0]["id"] == created["id"]


def test_manual_action_blocked_while_task_runs(mock_config, tmp_path, monkeypatch):
    c, app, _ = client(mock_config, tmp_path, monkeypatch, [step({"type": "wait", "seconds": 5})])
    ok = c.post("/api/devices/sim/action", json={"action": {"type": "home"}})
    assert ok.status_code == 200 and ok.json()["text"] == "回到桌面"
    task = c.post("/api/tasks", json={"device_id": "sim", "task": "等"}).json()
    time.sleep(0.2)
    busy = c.post("/api/devices/sim/action", json={"action": {"type": "back"}})
    assert busy.status_code == 409
    assert c.post("/api/devices/sim/action", json={"action": {"type": "tap", "point": [2000, 1]}}).status_code in (409, 422)
    assert c.post(f"/api/tasks/{task['id']}/cancel").json()["status"] == "cancelled"
    assert c.post("/api/tasks", json={"device_id": "sim", "task": "  "}).status_code == 409


def test_note_and_pause_resume(mock_config, tmp_path, monkeypatch):
    replies = [step({"type": "wait", "seconds": 1})] * 3 + [step({"type": "finish"})]
    c, _, model = client(mock_config, tmp_path, monkeypatch, replies)
    task = c.post("/api/tasks", json={"device_id": "sim", "task": "x"}).json()
    assert c.post(f"/api/tasks/{task['id']}/note", json={"text": "换成李四"}).status_code == 200
    assert c.post(f"/api/tasks/{task['id']}/pause").json()["status"] == "paused"
    time.sleep(0.3)
    assert c.post(f"/api/tasks/{task['id']}/resume", json={}).json()["status"] == "running"
    wait_status(c, task["id"], {"succeeded"})
    assert "换成李四" in model.requests[-1][1]["content"][0]["text"]


def test_qishui_plugin_only_executes_confirmed_back(mock_config, tmp_path, monkeypatch):
    replies = [step({"type": "tap", "point": [1, 1]}, reply="广告倒计时中"), step({"type": "back"}, reply="可以返回")]
    c, app, _ = client(mock_config, tmp_path, monkeypatch, replies)
    test = c.post("/api/plugins/qishui/start", json={"device_id": "sim"}).json()
    time.sleep(0.3)
    view = c.get(f"/api/plugins/qishui/{test['test_id']}").json()
    assert view["confirm"] is None and "点击" in view["proposal"]
    assert c.post(f"/api/plugins/qishui/{test['test_id']}/confirm", json={"mode": "back"}).status_code == 409
    c.post(f"/api/plugins/qishui/{test['test_id']}/observe")
    time.sleep(0.3)
    view = c.get(f"/api/plugins/qishui/{test['test_id']}").json()
    assert view["confirm"] == "back"
    done = c.post(f"/api/plugins/qishui/{test['test_id']}/confirm", json={"mode": "back"}).json()
    assert done["status"] == "completed"
    time.sleep(0.3)
    devices = app.state.devices
    executed = devices.get("sim").executions
    assert executed == [{"type": "back"}]
