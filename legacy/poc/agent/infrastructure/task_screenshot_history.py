"""Ordered, task-scoped screenshot evidence; never sample or scan other runs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time
from uuid import uuid4
from PIL import Image
from agent.domain.vision_model import VisionAgentError
from agent.infrastructure.atomic_files import atomic_replace_bytes, json_bytes

MANIFEST_NAME = "task_screenshots.json"


def cleanup_evidence_runs(root: Path, *, retention_seconds: float | None = None,
    max_bytes: int | None = None) -> dict[str, int]:
    """Remove only old completed evidence directories and enforce a disk budget.

    The model payload is never truncated. Cleanup happens between runs and skips
    directories carrying the ``.active`` marker so a live task keeps its full
    screenshot history.
    """

    base = Path(root)
    if not base.exists():
        return {"removed_runs": 0, "removed_bytes": 0}
    retention = float(retention_seconds if retention_seconds is not None else
        os.environ.get("ROBOT_EVIDENCE_RETENTION_SECONDS", 7 * 24 * 3600))
    budget = int(max_bytes if max_bytes is not None else
        os.environ.get("ROBOT_EVIDENCE_MAX_BYTES", 5 * 1024 * 1024 * 1024))
    now = time.time()
    candidates = []
    for path in base.iterdir():
        # Only directories created by the current evidence lifecycle are
        # eligible.  Unknown output folders and historical evidence are never
        # deleted by a generic startup cleanup.
        if (not path.is_dir() or not (path / ".evidence-run").exists()
                or (path / ".active").exists()):
            continue
        try:
            size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
            mtime = path.stat().st_mtime
        except OSError:
            continue
        candidates.append((mtime, path, size))
    removed_runs = removed_bytes = 0
    for mtime, path, size in sorted(candidates):
        if now - mtime < retention and sum(item[2] for item in candidates) - removed_bytes <= budget:
            continue
        try:
            shutil.rmtree(path)
        except OSError:
            continue
        removed_runs += 1
        removed_bytes += size
    return {"removed_runs": removed_runs, "removed_bytes": removed_bytes}


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
    quota = int(os.environ.get("ROBOT_EVIDENCE_MAX_RUN_BYTES", 1024 * 1024 * 1024))
    existing_bytes = sum(item.stat().st_size for item in directory.rglob("*") if item.is_file())
    estimated_bytes = sum(max(1024, frame.width * frame.height // 2) for frame in frames)
    if existing_bytes + estimated_bytes > quota:
        raise VisionAgentError(
            f"任务证据目录超过配额（{quota} bytes）；截图历史保持完整，未截断模型输入。"
        )
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
