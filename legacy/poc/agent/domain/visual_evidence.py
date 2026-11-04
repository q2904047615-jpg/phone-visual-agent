"""Pure evidence values produced by local visual measurements."""

from __future__ import annotations

from dataclasses import dataclass

from .validation import DataclassWire


@dataclass(frozen=True)
class LocalFrameStability:
    stable: bool
    mean_delta: float
    max_delta: float
    frame_count: int
    threshold: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {'stable': self.stable, 'mean_delta': round(self.mean_delta, 3), 'max_delta': round(self.max_delta, 3),
            'frame_count': self.frame_count, 'threshold': self.threshold, 'reason': self.reason}


@dataclass(frozen=True)
class VisualObstruction(DataclassWire):
    """A locally detected opaque region that can invalidate visual evidence."""

    kind: str
    bounds: tuple[int, int, int, int]
    reason: str
