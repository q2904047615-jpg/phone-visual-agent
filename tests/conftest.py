import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PVA_MOCK", "1")


class ScriptedModel:
    """Returns pre-written replies in order; records the messages it received."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def complete(self, messages, *, enable_thinking=False):
        from pva.model.qwen import ModelReply
        self.requests.append(messages)
        if not self.replies:
            raise AssertionError("ScriptedModel ran out of replies")
        reply = self.replies.pop(0)
        content = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        return ModelReply(content=content, usage={"total_tokens": 10})

    def status(self):
        return {"configured": True, "model": "scripted"}


def step(action, reply="", risk="none", last="none"):
    return {"observation": "obs", "last_result": last, "reply": reply, "reason": "r", "action": action, "risk": risk}


@pytest.fixture
def mock_config(tmp_path):
    from pva.config import load_config
    cfg = {"default_device": "sim", "agent": {"settle_seconds": 0},
           "devices": [{"id": "sim", "name": "模拟", "driver": "mock"},
                       {"id": "sim2", "name": "模拟2", "driver": "mock"}]}
    path = tmp_path / "devices.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    return load_config(path)
