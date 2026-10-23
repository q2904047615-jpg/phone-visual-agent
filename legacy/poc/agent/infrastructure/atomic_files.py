"""Durable same-directory file replacement shared by evidence adapters."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping
import uuid


def write_new_bytes(path: Path, payload: bytes) -> Path:
    target = Path(path)
    with target.open('xb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return target


def atomic_replace_bytes(path: Path, payload: bytes, *,
    replace_file: Callable[[Path, Path], None]=os.replace) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Same-directory atomic replacement without duplicating a potentially long basename.
    temporary = target.parent / f'.{uuid.uuid4().hex}.tmp'
    try:
        write_new_bytes(temporary, payload)
        replace_file(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def json_bytes(payload: Mapping[str, Any], *, trailing_newline: bool=True) -> bytes:
    suffix = '\n' if trailing_newline else ''
    return (json.dumps(dict(payload), ensure_ascii=False, indent=2) + suffix).encode('utf-8')
