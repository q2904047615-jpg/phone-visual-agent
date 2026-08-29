"""Atomic file-system persistence for authoritative typed-input lineage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

from PIL import Image

from agent.application.input_value_lineage import VerifiedInputActionType, build_surface_descriptors
from agent.domain.validation import reject_if
from agent.infrastructure.atomic_files import atomic_replace_bytes, json_bytes
from agent.domain.input_value_lineage import (
    DEFAULT_LINEAGE_TTL_SECONDS,
    InputValueLineageError,
    TypedInputLineage,
    build_verified_literal_lineage,
    build_verified_newline_lineage,
    build_verified_text_lineage,
)


class FileSystemTypedInputLineageStore:
    def __init__(self, directory: Path, *, ttl_seconds: float=DEFAULT_LINEAGE_TTL_SECONDS,
        clock: Any=time.time) -> None:
        self.directory = Path(directory)
        self.ttl_seconds = max(1.0, float(ttl_seconds))
        self.clock = clock

    def _path(self, device_id: str) -> Path:
        token = hashlib.sha256(device_id.encode("utf-8")).hexdigest()[:24]
        return self.directory / f"typed_input_lineage_{token}.json"

    def write(self, record: TypedInputLineage) -> Path:
        record.validate()
        destination = self._path(record.device_id)
        return atomic_replace_bytes(destination, json_bytes(record.to_dict()))

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
            record = TypedInputLineage.from_dict(json.loads(path.read_text(encoding='utf-8')))
        except (OSError, json.JSONDecodeError, InputValueLineageError):
            return None
        if record.device_id != device_id:
            return None
        now = float(self.clock())
        if now < record.recorded_at_epoch or now - record.recorded_at_epoch > self.ttl_seconds:
            return None
        return record

    def record_verified_action(self, *, action_type: VerifiedInputActionType, device_id: str,
        resolved_action: dict[str, Any], before_scene: dict[str, Any], after_scene: dict[str, Any],
        after_frames: tuple[Image.Image, ...], hardware_receipt: dict[str, Any] | None=None,
        source: str | None=None) -> TypedInputLineage:
        spec = {'literal': (build_verified_literal_lineage, 'verified_live_literal_action', True),
            'text': (build_verified_text_lineage, 'verified_live_text_action', False),
            'newline': (build_verified_newline_lineage, 'verified_live_newline_action', True)}.get(action_type)
        reject_if(spec is None, InputValueLineageError(f'未知 verified input action type：{action_type}'))
        builder, default_source, needs_receipt = spec
        values = {'device_id': device_id, 'resolved': resolved_action, 'before_scene': before_scene,
            'after_scene': after_scene, 'recorded_at_epoch': float(self.clock()), 'source': source or default_source,
            'surface_fallback': self.load(device_id),
            'surface_descriptor_factory': lambda bounds: build_surface_descriptors(after_frames, bounds)}
        if needs_receipt:
            values['hardware_receipt'] = hardware_receipt
        record = builder(**values)
        self.write(record)
        return record
