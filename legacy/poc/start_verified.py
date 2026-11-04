"""Start the API from this checkout after verifying its runtime wiring."""

from pathlib import Path
import hashlib
import inspect
import json
import os
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import web_app  # noqa: E402
from agent.infrastructure import robot_controller, seller_window_adapter  # noqa: E402


assert Path(web_app.__file__).resolve() == ROOT / "web_app.py"
assert not isinstance(web_app.runtime.controller, robot_controller.MockRobotController)
assert "seller_position" not in inspect.signature(robot_controller.RobotController).parameters
assert "select_camera_position" not in vars(seller_window_adapter)

vision_status = web_app.runtime.vision_provider.status()
assert vision_status["configured"]
assert vision_status["model"] == "qwen3-vl-plus"
assert vision_status["thinking_enabled"]

paths = [
    ROOT / "agent/infrastructure" / name
    for name in (
        "robot_controller.py",
        "seller_window_adapter.py",
        "device_controller_registry.py",
    )
]
print(
    json.dumps(
        {
            "pid": os.getpid(),
            "checkout": str(ROOT),
            "revision": web_app.runtime.loaded_code_revision,
            "sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths
            },
            "verified": True,
        },
        ensure_ascii=False,
    ),
    flush=True,
)

if "--check" not in sys.argv:
    import uvicorn

    uvicorn.run(web_app.app, host="127.0.0.1", port=8765)
