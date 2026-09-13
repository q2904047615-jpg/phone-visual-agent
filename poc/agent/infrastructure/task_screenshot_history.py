"""Ordered, task-scoped screenshot evidence; never sample or scan other runs."""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4
from PIL import Image
from agent.domain.vision_model import VisionAgentError
from agent.infrastructure.atomic_files import atomic_replace_bytes, json_bytes

MANIFEST_NAME = "task_screenshots.json"


def _manifest(directory: Path, device_id: str | None) -> dict:
    path = directory / MANIFEST_NAME
    value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "device_id": device_id, "task_id": None, "captures": []}
    if value["device_id"] != device_id:
        raise VisionAgentError("截图记录不属于当前设备。")
    return value


def save_task_frames(frames: list[Image.Image], directory: Path | None,
    prefix: str, device_id: str | None) -> tuple[str, ...]:
    if directory is None:
        return ()
    directory.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(directory, device_id)
    # Re-observing the same step must not overwrite any earlier capture.
    capture_id = uuid4().hex
    paths = []
    for index, frame in enumerate(frames, start=1):
        path = directory / f"{capture_id}_{index}.jpg"
        frame.save(path, format="JPEG", quality=92)
        paths.append(path)
    manifest["captures"].append({"capture": prefix, "files": [p.name for p in paths]})
    atomic_replace_bytes(directory / MANIFEST_NAME, json_bytes(manifest))
    return tuple(str(p) for p in paths)


def task_screenshots(directory: Path, *, device_id: str | None, task_id: str | None,
    current_paths: tuple[str, ...]) -> list[dict]:
    manifest = _manifest(directory, device_id)
    if manifest["task_id"] not in (None, task_id):
        raise VisionAgentError("截图记录不属于当前任务。")
    if manifest["task_id"] != task_id:
        manifest["task_id"] = task_id
        atomic_replace_bytes(directory / MANIFEST_NAME, json_bytes(manifest))
    result = []
    root = directory.resolve()
    for capture in manifest["captures"]:
        for name in capture["files"]:
            path = (root / name).resolve()
            if path.parent != root or path.suffix.lower() != ".jpg":
                raise VisionAgentError("任务截图路径超出本任务证据目录。")
            result.append({"capture": capture["capture"], "path": str(path)})
    current = [str(Path(p).resolve()) for p in current_paths]
    paths = [item["path"] for item in result]
    if not current or paths[-len(current):] != current or len(set(paths)) != len(paths):
        raise VisionAgentError("当前截图与任务截图记录不一致，不能提供完整任务证据。")
    for index, item in enumerate(result, start=1):
        item.update(image=index, group="CURRENT" if item["path"] in current else "HISTORY")
    return result
