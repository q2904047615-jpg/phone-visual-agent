"""Windows OCR subprocess adapter and deterministic text-box matching."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OCR_SCRIPT = ROOT / "windows_ocr.ps1"
POWERSHELL = Path(
    os.environ.get(
        "SystemRoot", r"C:\Windows"
    )
) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"


class OcrUnavailableError(RuntimeError):
    pass


class OcrRecognitionError(RuntimeError):
    pass


@dataclass(frozen=True)
class OcrMatch:
    text: str
    left: int
    top: int
    width: int
    height: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.left + self.width // 2, self.top + self.height // 2)


def is_available() -> bool:
    return os.name == "nt" and POWERSHELL.is_file() and OCR_SCRIPT.is_file()


def _compact(text: str) -> str:
    return re.sub(r"[\s\u3000]+", "", text or "").lower()


def _match_from_items(text: str, items: list[dict[str, Any]]) -> OcrMatch:
    left = min(float(item.get("left", 0)) for item in items)
    top = min(float(item.get("top", 0)) for item in items)
    right = max((float(item.get('left', 0)) + float(item.get('width', 1)) for item in items))
    bottom = max((float(item.get('top', 0)) + float(item.get('height', 1)) for item in items))
    return OcrMatch(
        text=text,
        left=int(round(left)),
        top=int(round(top)),
        width=max(1, int(round(right - left))),
        height=max(1, int(round(bottom - top))),
    )


def find_text(payload: dict[str, Any], target: str) -> list[OcrMatch]:
    needle = _compact(target)
    if not needle:
        return []
    found: list[OcrMatch] = []
    seen: set[tuple[int, int, int, int]] = set()

    for line in payload.get("lines") or []:
        words = list(line.get("words") or [])
        compact_words = [_compact(str(word.get("text", ""))) for word in words]
        combined = "".join(compact_words)
        start = combined.find(needle)
        match: OcrMatch | None = None
        if start >= 0 and words:
            stop = start + len(needle)
            cursor = 0
            selected: list[dict[str, Any]] = []
            for word, word_text in zip(words, compact_words):
                word_stop = cursor + len(word_text)
                if word_stop > start and cursor < stop:
                    selected.append(word)
                cursor = word_stop
            if selected:
                match = _match_from_items(target, selected)
        elif needle in _compact(str(line.get("text", ""))):
            match = _match_from_items(target, [line])

        if match is None:
            continue
        box = (match.left, match.top, match.width, match.height)
        if box in seen:
            continue
        seen.add(box)
        found.append(match)
    return found


def _rescale_box(item: dict[str, Any], scale: float) -> None:
    for key in ("left", "top", "width", "height"):
        if key in item:
            item[key] = round(float(item[key]) / scale, 2)
    for word in item.get("words") or []:
        _rescale_box(word, scale)


def recognize( image: Image.Image, language: str = "zh-Hans-CN", *, scale: float = 3.0, ) -> dict[str, Any]:
    if not is_available():
        raise OcrUnavailableError("Windows 简体中文 OCR 不可用。")

    path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            path = handle.name
        source = image.convert("RGB")
        actual_scale = max(1.0, float(scale))
        if actual_scale > 1.0:
            source = source.resize(
                (
                    max(1, int(round(source.width * actual_scale))),
                    max(1, int(round(source.height * actual_scale))),
                ),
                Image.Resampling.LANCZOS,
            )
        source.save(path, format="PNG")
        process = subprocess.run(
            [
                str(POWERSHELL),
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(OCR_SCRIPT),
                "-ImagePath",
                path,
                "-LanguageTag",
                language,
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=20,
            check=False,
        )
        stdout = process.stdout.decode("utf-8-sig", errors="replace").strip()
        stderr = process.stderr.decode("utf-8-sig", errors="replace").strip()
        if process.returncode != 0:
            raise OcrRecognitionError(f'Windows OCR 执行失败（{process.returncode}）：{stderr or stdout}')
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise OcrRecognitionError(f'Windows OCR 返回内容无法解析：{stdout[:300]}') from exc
        if not isinstance(payload, dict):
            raise OcrRecognitionError("Windows OCR 返回格式错误。")
        if actual_scale > 1.0:
            for line in payload.get("lines") or []:
                _rescale_box(line, actual_scale)
        return payload
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass
