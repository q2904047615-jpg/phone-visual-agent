"""Atomic file-system persistence for authoritative typed-input lineage."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from PIL import Image

from agent.application.input_value_lineage import build_surface_descriptors
from agent.domain.input_value_lineage import (
    DEFAULT_LINEAGE_TTL_SECONDS,
    InputValueLineageError,
    TypedInputLineage,
    build_verified_literal_lineage,
    build_verified_newline_lineage,
    build_verified_text_lineage,
)


class FileSystemTypedInputLineageStore:
    def __init__(
        self,
        directory: Path,
        *,
        ttl_seconds: float = DEFAULT_LINEAGE_TTL_SECONDS,
        clock: Any = time.time,
    ) -> None:
        self.directory = Path(directory)
        self.ttl_seconds = max(1.0, float(ttl_seconds))
        self.clock = clock

    def _path(self, device_id: str) -> Path:
        token = hashlib.sha256(device_id.encode("utf-8")).hexdigest()[:24]
        return self.directory / f"typed_input_lineage_{token}.json"

    def write(self, record: TypedInputLineage) -> Path:
        record.validate()
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self._path(record.device_id)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.stem}.",
            suffix=".tmp",
            dir=str(self.directory),
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(record.to_dict(), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        return destination

    def discard(self, device_id: str) -> None:
        """Remove only the stale non-empty value for one verified device."""

        try:
            self._path(device_id).unlink()
        except FileNotFoundError:
            return

    def load(self, device_id: str) -> TypedInputLineage | None:
        path = self._path(device_id)
        if not path.is_file():
            return None
        try:
            record = TypedInputLineage.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, InputValueLineageError):
            return None
        if record.device_id != device_id:
            return None
        now = float(self.clock())
        if now < record.recorded_at_epoch or now - record.recorded_at_epoch > self.ttl_seconds:
            return None
        return record

    @staticmethod
    def _descriptor_factory(
        after_frames: tuple[Image.Image, ...],
    ) -> Any:
        return lambda bounds: build_surface_descriptors(after_frames, bounds)

    def record_verified_literal_action(
        self,
        *,
        device_id: str,
        resolved_action: dict[str, Any],
        before_scene: dict[str, Any],
        after_scene: dict[str, Any],
        hardware_receipt: dict[str, Any],
        after_frames: tuple[Image.Image, ...],
        source: str = "verified_live_literal_action",
    ) -> TypedInputLineage:
        record = build_verified_literal_lineage(
            device_id=device_id,
            resolved=resolved_action,
            before_scene=before_scene,
            after_scene=after_scene,
            hardware_receipt=hardware_receipt,
            recorded_at_epoch=float(self.clock()),
            source=source,
            surface_fallback=self.load(device_id),
            surface_descriptor_factory=self._descriptor_factory(after_frames),
        )
        self.write(record)
        return record

    def record_verified_text_action(
        self,
        *,
        device_id: str,
        resolved_action: dict[str, Any],
        before_scene: dict[str, Any],
        after_scene: dict[str, Any],
        after_frames: tuple[Image.Image, ...],
        source: str = "verified_live_text_action",
    ) -> TypedInputLineage:
        record = build_verified_text_lineage(
            device_id=device_id,
            resolved=resolved_action,
            before_scene=before_scene,
            after_scene=after_scene,
            recorded_at_epoch=float(self.clock()),
            source=source,
            surface_fallback=self.load(device_id),
            surface_descriptor_factory=self._descriptor_factory(after_frames),
        )
        self.write(record)
        return record

    def record_verified_newline_action(
        self,
        *,
        device_id: str,
        resolved_action: dict[str, Any],
        before_scene: dict[str, Any],
        after_scene: dict[str, Any],
        hardware_receipt: dict[str, Any],
        after_frames: tuple[Image.Image, ...],
        source: str = "verified_live_newline_action",
    ) -> TypedInputLineage:
        record = build_verified_newline_lineage(
            device_id=device_id,
            resolved=resolved_action,
            before_scene=before_scene,
            after_scene=after_scene,
            hardware_receipt=hardware_receipt,
            recorded_at_epoch=float(self.clock()),
            source=source,
            surface_fallback=self.load(device_id),
            surface_descriptor_factory=self._descriptor_factory(after_frames),
        )
        self.write(record)
        return record
