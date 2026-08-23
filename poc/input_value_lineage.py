from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


TYPED_INPUT_LINEAGE_VERSION = "2026-08-23-typed-input-lineage-v4"
DEFAULT_LINEAGE_TTL_SECONDS = 6 * 60 * 60
SURFACE_DESCRIPTOR_WIDTH = 32
SURFACE_DESCRIPTOR_HEIGHT = 16
SURFACE_DESCRIPTOR_MAX_MEAN_DISTANCE = 18.0
PENDING_INPUT_LINEAGE_SOURCES = frozenset(
    {
        "pending_verified_literal_action",
        "pending_verified_text_action",
        "pending_verified_input_state_action",
        "pending_verified_ime_candidate_action",
        "pending_verified_newline_action",
    }
)
NEWLINE_INPUT_LINEAGE_SOURCES = frozenset(
    {
        "pending_verified_newline_action",
        "pending_verified_text_action",
        "verified_live_newline_action",
        "verified_live_text_action",
        "verified_persisted_newline_execution",
        "verified_persisted_text_execution",
    }
)


class InputValueLineageError(ValueError):
    pass


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _valid_bounds(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return None
    bounds = tuple(float(item) for item in value)
    left, top, right, bottom = bounds
    if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
        return None
    return bounds


def _bounds_compatible(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    coverage = intersection / min(first_area, second_area)
    first_center = ((first[0] + first[2]) / 2, (first[1] + first[3]) / 2)
    second_center = ((second[0] + second[2]) / 2, (second[1] + second[3]) / 2)
    return bool(
        coverage >= 0.25
        and abs(first_center[0] - second_center[0]) <= 0.10
        and abs(first_center[1] - second_center[1]) <= 0.08
    )


def _collapsed_visual_text(value: str) -> str:
    return value.replace("\r", "").replace("\n", "")


def _exact_or_soft_wrapped_visual_text(raw_value: str, exact_value: str) -> bool:
    """Compare exact values without turning a real newline into soft wrap.

    A stored value that contains an authorized newline must be observed exactly.
    Collapsing visual rows remains valid only for single-line exact values.
    """

    if "\r" in exact_value or "\n" in exact_value:
        return raw_value == exact_value
    return _collapsed_visual_text(raw_value) == exact_value


def _surface_descriptor(
    frame: Image.Image,
    bounds: tuple[float, float, float, float],
) -> str:
    if not isinstance(frame, Image.Image):
        raise InputValueLineageError("输入表面描述缺少真实图像帧。")
    valid = _valid_bounds(bounds)
    if valid is None or frame.width < 2 or frame.height < 2:
        raise InputValueLineageError("输入表面描述的图像或 bounds 无效。")
    left, top, right, bottom = valid
    left = max(0.0, left - 0.04)
    top = max(0.0, top - 0.035)
    right = min(1.0, right + 0.04)
    bottom = min(1.0, bottom + 0.035)
    pixel_box = (
        round(left * frame.width),
        round(top * frame.height),
        round(right * frame.width),
        round(bottom * frame.height),
    )
    if pixel_box[0] >= pixel_box[2] or pixel_box[1] >= pixel_box[3]:
        raise InputValueLineageError("输入表面描述的局部区域为空。")
    gray = frame.convert("L").crop(pixel_box)
    normalized = ImageOps.autocontrast(gray, cutoff=1).resize(
        (SURFACE_DESCRIPTOR_WIDTH, SURFACE_DESCRIPTOR_HEIGHT),
        Image.Resampling.LANCZOS,
    )
    return normalized.tobytes().hex()


def _valid_surface_descriptor(value: Any) -> bool:
    expected_length = SURFACE_DESCRIPTOR_WIDTH * SURFACE_DESCRIPTOR_HEIGHT * 2
    return bool(
        isinstance(value, str)
        and len(value) == expected_length
        and all(character in "0123456789abcdef" for character in value)
    )


def _surface_descriptors(
    frames: Any,
    bounds: tuple[float, float, float, float],
) -> tuple[str, ...]:
    if not isinstance(frames, (list, tuple)) or len(frames) != 4:
        raise InputValueLineageError("输入表面连续性必须绑定动作后四帧。")
    descriptors = tuple(_surface_descriptor(frame, bounds) for frame in frames)
    if len(set(descriptors)) == 0:
        raise InputValueLineageError("输入表面连续性没有可用的局部描述。")
    return descriptors


def _surface_descriptor_matches(
    descriptors: tuple[str, ...],
    *,
    frame: Image.Image | None,
    bounds: tuple[float, float, float, float],
) -> bool:
    if frame is None or not descriptors:
        return False
    try:
        current = bytes.fromhex(_surface_descriptor(frame, bounds))
    except (InputValueLineageError, ValueError):
        return False
    for descriptor in descriptors:
        try:
            prior = bytes.fromhex(descriptor)
        except ValueError:
            continue
        if len(prior) != len(current):
            continue
        mean_distance = sum(
            abs(first - second) for first, second in zip(prior, current)
        ) / len(current)
        if mean_distance <= SURFACE_DESCRIPTOR_MAX_MEAN_DISTANCE:
            return True
    return False


def _surface_identity_compatible(
    *,
    recorded_app_id: str,
    recorded_screen_id: str,
    current_app_id: str,
    current_screen_id: str,
    exact_value: str,
) -> bool:
    current_app = str(current_app_id or "").strip()
    current_screen = str(current_screen_id or "").strip()
    if not current_screen or current_screen == "unknown":
        return False
    if current_app not in {recorded_app_id, "unknown"}:
        return False
    screen_related = bool(
        current_screen == recorded_screen_id
        or current_screen.startswith(recorded_screen_id + "_")
        or recorded_screen_id.startswith(current_screen + "_")
    )
    if not screen_related:
        return False
    if current_app == recorded_app_id:
        return True
    # An unknown visual App label cannot replace a known identity. It may only
    # preserve a sufficiently distinctive exact-value chain on the same
    # semantically related input surface.
    return len(exact_value) >= 8


@dataclass(frozen=True)
class TypedInputLineage:
    version: str
    device_id: str
    exact_value: str
    app_id: str
    screen_id: str
    input_meaning: str
    input_field_id: str
    input_bounds: tuple[float, float, float, float]
    before_fingerprint: str
    after_fingerprint: str
    action_digest: str
    receipt_digest: str
    surface_descriptors: tuple[str, ...]
    recorded_at_epoch: float
    source: str

    def validate(self) -> None:
        if self.version != TYPED_INPUT_LINEAGE_VERSION:
            raise InputValueLineageError("输入值连续性协议版本不匹配。")
        for name in (
            "device_id",
            "app_id",
            "screen_id",
            "input_meaning",
            "input_field_id",
            "before_fingerprint",
            "after_fingerprint",
            "action_digest",
            "receipt_digest",
            "source",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise InputValueLineageError(f"输入值连续性缺少 {name}。")
        if self.input_meaning != "application_text_input":
            raise InputValueLineageError("输入值连续性只接受应用文字输入框。")
        if not isinstance(self.exact_value, str) or not self.exact_value:
            raise InputValueLineageError("输入值连续性 exact_value 无效。")
        if "\r" in self.exact_value:
            raise InputValueLineageError("输入值连续性不接受回车字符。")
        if "\n" in self.exact_value and self.source not in NEWLINE_INPUT_LINEAGE_SOURCES:
            raise InputValueLineageError(
                "只有经过严格换行动作链验证的连续性才能保存真实换行。"
            )
        if _valid_bounds(self.input_bounds) is None:
            raise InputValueLineageError("输入值连续性 input_bounds 无效。")
        if self.before_fingerprint == self.after_fingerprint:
            raise InputValueLineageError("输入值连续性前后 fingerprint 未变化。")
        for digest_name in ("action_digest", "receipt_digest"):
            digest = getattr(self, digest_name)
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise InputValueLineageError(f"输入值连续性 {digest_name} 无效。")
        if not isinstance(self.surface_descriptors, tuple) or any(
            not _valid_surface_descriptor(item) for item in self.surface_descriptors
        ):
            raise InputValueLineageError("输入值连续性的局部画面描述无效。")
        if (
            self.source not in PENDING_INPUT_LINEAGE_SOURCES
            and len(self.surface_descriptors) != 4
        ):
            raise InputValueLineageError("持久输入值连续性必须绑定动作后四帧。")
        if self.source in PENDING_INPUT_LINEAGE_SOURCES and self.surface_descriptors:
            raise InputValueLineageError("临时输入值连续性不能伪造持久画面描述。")
        if isinstance(self.recorded_at_epoch, bool) or not isinstance(
            self.recorded_at_epoch, (int, float)
        ):
            raise InputValueLineageError("输入值连续性时间无效。")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["input_bounds"] = list(self.input_bounds)
        value["surface_descriptors"] = list(self.surface_descriptors)
        return value

    @classmethod
    def from_dict(cls, value: Any) -> "TypedInputLineage":
        if not isinstance(value, dict):
            raise InputValueLineageError("输入值连续性必须是 JSON 对象。")
        required = {
            "version",
            "device_id",
            "exact_value",
            "app_id",
            "screen_id",
            "input_meaning",
            "input_field_id",
            "input_bounds",
            "before_fingerprint",
            "after_fingerprint",
            "action_digest",
            "receipt_digest",
            "surface_descriptors",
            "recorded_at_epoch",
            "source",
        }
        if set(value) != required:
            raise InputValueLineageError("输入值连续性字段不完整或包含额外字段。")
        bounds = _valid_bounds(value.get("input_bounds"))
        if bounds is None:
            raise InputValueLineageError("输入值连续性 input_bounds 无效。")
        record = cls(
            version=value["version"],
            device_id=value["device_id"],
            exact_value=value["exact_value"],
            app_id=value["app_id"],
            screen_id=value["screen_id"],
            input_meaning=value["input_meaning"],
            input_field_id=value["input_field_id"],
            input_bounds=bounds,
            before_fingerprint=value["before_fingerprint"],
            after_fingerprint=value["after_fingerprint"],
            action_digest=value["action_digest"],
            receipt_digest=value["receipt_digest"],
            surface_descriptors=tuple(value["surface_descriptors"])
            if isinstance(value.get("surface_descriptors"), list)
            else (),
            recorded_at_epoch=value["recorded_at_epoch"],
            source=value["source"],
        )
        record.validate()
        return record

    def matches_visual(
        self,
        *,
        device_id: str,
        app_id: str,
        screen_id: str,
        raw_value: str,
        input_bounds: tuple[float, float, float, float] | None = None,
        now_epoch: float | None = None,
        ttl_seconds: float = DEFAULT_LINEAGE_TTL_SECONDS,
        current_frame: Image.Image | None = None,
    ) -> bool:
        now = time.time() if now_epoch is None else float(now_epoch)
        if (
            device_id != self.device_id
            or now < self.recorded_at_epoch
            or now - self.recorded_at_epoch > ttl_seconds
            or not isinstance(raw_value, str)
            or not ({"\r", "\n"} & set(raw_value))
            or not _exact_or_soft_wrapped_visual_text(raw_value, self.exact_value)
        ):
            return False
        if input_bounds is not None and not _bounds_compatible(
            self.input_bounds, input_bounds
        ):
            return False
        if _surface_identity_compatible(
            recorded_app_id=self.app_id,
            recorded_screen_id=self.screen_id,
            current_app_id=app_id,
            current_screen_id=screen_id,
            exact_value=self.exact_value,
        ):
            return True
        return _surface_descriptor_matches(
            self.surface_descriptors,
            frame=current_frame,
            bounds=self.input_bounds,
        )

    def matches_persisted_surface(
        self,
        *,
        device_id: str,
        app_id: str,
        screen_id: str,
        input_bounds: tuple[float, float, float, float] | None,
        current_frame: Image.Image | None,
        now_epoch: float | None = None,
        ttl_seconds: float = DEFAULT_LINEAGE_TTL_SECONDS,
    ) -> bool:
        """Rebind an immutable persisted value to the same live input surface.

        This method deliberately does not inspect or return a model-transcribed
        value.  It only proves that a prior four-frame action receipt still
        describes the current App/screen/input crop.  A caller must separately
        require an exact visible cue before using ``exact_value``.
        """

        now = time.time() if now_epoch is None else float(now_epoch)
        return bool(
            self.source not in PENDING_INPUT_LINEAGE_SOURCES
            and device_id == self.device_id
            and now >= self.recorded_at_epoch
            and now - self.recorded_at_epoch <= ttl_seconds
            and input_bounds is not None
            and _bounds_compatible(self.input_bounds, input_bounds)
            and _surface_identity_compatible(
                recorded_app_id=self.app_id,
                recorded_screen_id=self.screen_id,
                current_app_id=app_id,
                current_screen_id=screen_id,
                exact_value=self.exact_value,
            )
            and _surface_descriptor_matches(
                self.surface_descriptors,
                frame=current_frame,
                bounds=input_bounds,
            )
        )

    def matches_persisted_surface_cue(
        self,
        *,
        device_id: str,
        app_id: str,
        screen_id: str,
        raw_value: str,
        visible_editable_cues: tuple[str, ...],
        input_bounds: tuple[float, float, float, float] | None,
        current_frame: Image.Image | None,
        now_epoch: float | None = None,
        ttl_seconds: float = DEFAULT_LINEAGE_TTL_SECONDS,
    ) -> bool:
        """Recover only one exact cue on a revalidated persisted surface."""

        return bool(
            raw_value == ""
            and tuple(visible_editable_cues).count(self.exact_value) == 1
            and self.matches_persisted_surface(
                device_id=device_id,
                app_id=app_id,
                screen_id=screen_id,
                input_bounds=input_bounds,
                current_frame=current_frame,
                now_epoch=now_epoch,
                ttl_seconds=ttl_seconds,
            )
        )

    def matches_pending_input_state_value(
        self,
        *,
        device_id: str,
        app_id: str,
        screen_id: str,
        raw_value: str,
        input_bounds: tuple[float, float, float, float] | None,
        input_field_id: str | None = None,
        now_epoch: float | None = None,
        ttl_seconds: float = DEFAULT_LINEAGE_TTL_SECONDS,
    ) -> bool:
        """Bind an immediate state-only keyboard action to the same input."""

        return bool(
            isinstance(raw_value, str)
            and raw_value == self.exact_value
            and self.matches_pending_input_state_surface(
                device_id=device_id,
                app_id=app_id,
                screen_id=screen_id,
                input_bounds=input_bounds,
                input_field_id=input_field_id,
                now_epoch=now_epoch,
                ttl_seconds=ttl_seconds,
            )
        )

    def matches_pending_input_state_surface(
        self,
        *,
        device_id: str,
        app_id: str,
        screen_id: str,
        input_bounds: tuple[float, float, float, float] | None,
        input_field_id: str | None = None,
        now_epoch: float | None = None,
        ttl_seconds: float = DEFAULT_LINEAGE_TTL_SECONDS,
    ) -> bool:
        """Authorize an immediate exact visual-cue check on the same input."""

        now = time.time() if now_epoch is None else float(now_epoch)
        current_screen = str(screen_id or "").strip().casefold()
        recorded_screen = self.screen_id.strip().casefold()
        current_field = str(input_field_id or "").strip()
        typed_field_matches = bool(
            self.input_field_id not in {"", "unknown"}
            and current_field == self.input_field_id
        )
        return bool(
            self.source in {
                "pending_verified_input_state_action",
                "pending_verified_ime_candidate_action",
                "pending_verified_newline_action",
            }
            and device_id == self.device_id
            and now >= self.recorded_at_epoch
            and now - self.recorded_at_epoch <= ttl_seconds
            and str(app_id or "").strip().casefold()
            == self.app_id.strip().casefold()
            and current_screen
            and (
                current_screen == recorded_screen
                or current_screen.startswith(recorded_screen + "_")
                or recorded_screen.startswith(current_screen + "_")
                or typed_field_matches
            )
            and input_bounds is not None
            and _bounds_compatible(self.input_bounds, input_bounds)
        )

    def matches_pending_input_state_cue(
        self,
        *,
        device_id: str,
        app_id: str,
        screen_id: str,
        raw_value: str,
        visible_editable_cues: tuple[str, ...],
        input_bounds: tuple[float, float, float, float] | None,
        input_field_id: str | None = None,
        now_epoch: float | None = None,
        ttl_seconds: float = DEFAULT_LINEAGE_TTL_SECONDS,
    ) -> bool:
        """Recover only an exact value still visible inside the same input."""

        return bool(
            raw_value == ""
            and tuple(visible_editable_cues).count(self.exact_value) == 1
            and self.matches_pending_input_state_value(
                device_id=device_id,
                app_id=app_id,
                screen_id=screen_id,
                raw_value=self.exact_value,
                input_bounds=input_bounds,
                input_field_id=input_field_id,
                now_epoch=now_epoch,
                ttl_seconds=ttl_seconds,
            )
        )

    def pending_text_committed_prefix(
        self,
        *,
        device_id: str,
        app_id: str,
        screen_id: str,
        authorized_text: str,
        coarse_exact_value: str,
        raw_value: str,
        preedit_text: str,
        input_bounds: tuple[float, float, float, float] | None,
        input_field_id: str | None,
        now_epoch: float | None = None,
        ttl_seconds: float = DEFAULT_LINEAGE_TTL_SECONDS,
    ) -> str | None:
        """Recover only the committed prefix hidden beside an IME preedit.

        A returned direct-text action may leave its new fragment in the IME
        composition buffer.  Some dedicated audits then report that preedit
        correctly but omit the already committed prefix after the placeholder
        disappears.  The prefix is derivable only when the pending typed
        lineage, the exact authorized payload, the independent coarse read,
        the same field id and the same input surface all agree.  The preedit
        itself remains uncommitted and must still be selected separately.
        """

        now = time.time() if now_epoch is None else float(now_epoch)
        current_field = str(input_field_id or "").strip()
        if (
            self.source != "pending_verified_text_action"
            or device_id != self.device_id
            or now < self.recorded_at_epoch
            or now - self.recorded_at_epoch > ttl_seconds
            or authorized_text != self.exact_value
            or coarse_exact_value != self.exact_value
            or raw_value != ""
            or not isinstance(preedit_text, str)
            or not preedit_text
            or not self.exact_value.endswith(preedit_text)
            or self.exact_value == preedit_text
            or self.input_field_id in {"", "unknown"}
            or current_field != self.input_field_id
            or input_bounds is None
            or not _bounds_compatible(self.input_bounds, input_bounds)
            or not _surface_identity_compatible(
                recorded_app_id=self.app_id,
                recorded_screen_id=self.screen_id,
                current_app_id=app_id,
                current_screen_id=screen_id,
                exact_value=self.exact_value,
            )
        ):
            return None
        committed_prefix = self.exact_value[: -len(preedit_text)]
        return committed_prefix if committed_prefix and "\r" not in committed_prefix else None

    def matches_trailing_newline_cue(
        self,
        *,
        device_id: str,
        app_id: str,
        screen_id: str,
        raw_value: str,
        visible_editable_cues: tuple[str, ...],
        caret_line_index: int | None,
        input_bounds: tuple[float, float, float, float] | None,
        input_field_id: str | None,
        current_frame: Image.Image | None = None,
        now_epoch: float | None = None,
        ttl_seconds: float = DEFAULT_LINEAGE_TTL_SECONDS,
    ) -> bool:
        """Prove one trailing newline from action lineage plus caret geometry.

        Visual row layout alone is never newline authority.  This matcher only
        applies to a lineage minted by an exact ``press_enter`` action and then
        requires the same typed field, the exact visible prior value, and the
        caret on the newly created zero-based line.
        """

        if (
            self.source not in NEWLINE_INPUT_LINEAGE_SOURCES
            or not self.exact_value.endswith("\n")
            or isinstance(caret_line_index, bool)
            or not isinstance(caret_line_index, int)
            or caret_line_index != self.exact_value.count("\n")
        ):
            return False
        prior = self.exact_value[:-1]
        decorative = {
            "border",
            "caret",
            "cursor",
            "focus border",
            "focus ring",
            "outline",
            "|",
        }
        literal_cues = tuple(
            cue
            for cue in visible_editable_cues
            if isinstance(cue, str)
            and cue.strip()
            and cue.strip().casefold() not in decorative
        )
        if not (
            raw_value == prior
            or (raw_value == "" and literal_cues == (prior,))
        ):
            return False
        if (
            self.input_field_id in {"", "unknown"}
            or input_field_id != self.input_field_id
        ):
            return False
        if self.source in PENDING_INPUT_LINEAGE_SOURCES:
            return self.matches_pending_input_state_surface(
                device_id=device_id,
                app_id=app_id,
                screen_id=screen_id,
                input_bounds=input_bounds,
                input_field_id=input_field_id,
                now_epoch=now_epoch,
                ttl_seconds=ttl_seconds,
            )
        return bool(
            self.input_field_id not in {"", "unknown"}
            and input_field_id == self.input_field_id
            and self.matches_persisted_surface(
                device_id=device_id,
                app_id=app_id,
                screen_id=screen_id,
                input_bounds=input_bounds,
                current_frame=current_frame,
                now_epoch=now_epoch,
                ttl_seconds=ttl_seconds,
            )
        )


class TypedInputLineageStore:
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


    def match_visual(
        self,
        *,
        device_id: str,
        app_id: str,
        screen_id: str,
        raw_value: str | None,
        input_bounds: tuple[float, float, float, float] | None = None,
        current_frame: Image.Image | None = None,
    ) -> TypedInputLineage | None:
        record = self.load(device_id)
        if record is None or not isinstance(raw_value, str):
            return None
        if not record.matches_visual(
            device_id=device_id,
            app_id=app_id,
            screen_id=screen_id,
            raw_value=raw_value,
            input_bounds=input_bounds,
            now_epoch=float(self.clock()),
            ttl_seconds=self.ttl_seconds,
            current_frame=current_frame,
        ):
            return None
        return record

    def match_surface(
        self,
        *,
        device_id: str,
        app_id: str,
        screen_id: str,
        input_bounds: tuple[float, float, float, float] | None,
        current_frame: Image.Image | None,
    ) -> TypedInputLineage | None:
        record = self.load(device_id)
        if record is None or not record.matches_persisted_surface(
            device_id=device_id,
            app_id=app_id,
            screen_id=screen_id,
            input_bounds=input_bounds,
            current_frame=current_frame,
            now_epoch=float(self.clock()),
            ttl_seconds=self.ttl_seconds,
        ):
            return None
        return record

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
        record = _record_from_execution(
            device_id=device_id,
            resolved=resolved_action,
            before_scene=before_scene,
            after_scene=after_scene,
            hardware_receipt=hardware_receipt,
            recorded_at_epoch=float(self.clock()),
            source=source,
            surface_fallback=self.load(device_id),
            surface_frames=after_frames,
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
        record = _record_from_text_execution(
            device_id=device_id,
            resolved=resolved_action,
            before_scene=before_scene,
            after_scene=after_scene,
            recorded_at_epoch=float(self.clock()),
            source=source,
            surface_fallback=self.load(device_id),
            surface_frames=after_frames,
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
        record = _record_from_newline_execution(
            device_id=device_id,
            resolved=resolved_action,
            before_scene=before_scene,
            after_scene=after_scene,
            hardware_receipt=hardware_receipt,
            recorded_at_epoch=float(self.clock()),
            source=source,
            surface_fallback=self.load(device_id),
            surface_frames=after_frames,
        )
        self.write(record)
        return record

    def recover_from_session_file(self, path: Path) -> TypedInputLineage:
        session_path = Path(path)
        payload = json.loads(session_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise InputValueLineageError("历史 session 不是 JSON 对象。")
        device_id = payload.get("device_id")
        history = payload.get("history")
        if not isinstance(device_id, str) or not device_id.strip():
            raise InputValueLineageError("历史 session 缺少 device_id。")
        if not isinstance(history, list) or not history:
            raise InputValueLineageError("历史 session 没有执行记录。")
        execution = history[-1].get("execution") if isinstance(history[-1], dict) else None
        if not isinstance(execution, dict) or execution.get("physical_actions") != 1:
            raise InputValueLineageError("历史执行没有且仅有一次物理动作。")
        before_paths = execution.get("before_frame_paths")
        after_paths = execution.get("after_frame_paths")
        if (
            not isinstance(before_paths, list)
            or len(before_paths) != 4
            or not isinstance(after_paths, list)
            or len(after_paths) != 4
            or any(not isinstance(item, str) or not Path(item).is_file() for item in before_paths + after_paths)
        ):
            raise InputValueLineageError("历史执行缺少完整的动作前后四帧文件。")
        loaded_after_frames: list[Image.Image] = []
        for item in after_paths:
            try:
                with Image.open(item) as image:
                    loaded_after_frames.append(image.copy())
            except (OSError, ValueError) as exc:
                raise InputValueLineageError(
                    "历史执行的动作后帧无法解码。"
                ) from exc
        resolved_action = execution.get("resolved_action")
        if (
            isinstance(resolved_action, dict)
            and resolved_action.get("kind") == "input_verified_text"
        ):
            if (
                execution.get("action_outcome") != "matched"
                or execution.get("verification_errors") not in (None, [], ())
            ):
                raise InputValueLineageError(
                    "历史文字分段没有通过动作后 exact verifier。"
                )
            record = _record_from_text_execution(
                device_id=device_id,
                resolved=resolved_action,
                before_scene=execution.get("before_scene"),
                after_scene=execution.get("after_scene"),
                recorded_at_epoch=float(self.clock()),
                source="verified_persisted_text_execution",
                surface_frames=tuple(loaded_after_frames),
            )
        elif (
            isinstance(resolved_action, dict)
            and resolved_action.get("kind") == "press_enter"
        ):
            if (
                execution.get("action_outcome") != "matched"
                or execution.get("verification_errors") not in (None, [], ())
            ):
                raise InputValueLineageError(
                    "历史换行动作没有通过动作后 exact verifier。"
                )
            record = _record_from_newline_execution(
                device_id=device_id,
                resolved=resolved_action,
                before_scene=execution.get("before_scene"),
                after_scene=execution.get("after_scene"),
                hardware_receipt=execution.get("hardware_receipt"),
                recorded_at_epoch=float(self.clock()),
                source="verified_persisted_newline_execution",
                surface_frames=tuple(loaded_after_frames),
            )
        else:
            record = _record_from_execution(
                device_id=device_id,
                resolved=resolved_action,
                before_scene=execution.get("before_scene"),
                after_scene=execution.get("after_scene"),
                hardware_receipt=execution.get("hardware_receipt"),
                recorded_at_epoch=float(self.clock()),
                source="verified_persisted_literal_execution",
                surface_frames=tuple(loaded_after_frames),
            )
        self.write(record)
        return record


def build_pending_text_lineage(
    *,
    device_id: str,
    resolved_action: dict[str, Any],
    before_scene: dict[str, Any],
    recorded_at_epoch: float | None = None,
) -> TypedInputLineage:
    """Bind one returned text transaction to its immediate visual result."""

    parts = _validated_text_action_chain(resolved_action, before_scene)
    before_input, prior, expected, _fragment = parts
    app_id = before_scene.get("app_id")
    screen_id = before_scene.get("screen_id")
    before_fingerprint = before_scene.get("fingerprint")
    input_field_id = _typed_input_field_id(before_input)
    if (
        not isinstance(app_id, str)
        or not app_id.strip()
        or not isinstance(screen_id, str)
        or not screen_id.strip()
        or screen_id == "unknown"
        or not isinstance(before_fingerprint, str)
        or not before_fingerprint.strip()
        or before_fingerprint == "unknown"
        or (app_id == "unknown" and input_field_id == "unknown")
    ):
        raise InputValueLineageError("临时文字连续性缺少明确输入表面。")
    action_digest = _canonical_digest(resolved_action)
    receipt_digest = _canonical_digest(
        {
            "protocol_version": "2026-08-20-verified-text-transaction-v1",
            "stage": "controller_call_returned",
            "device_id": device_id,
            "action_digest": action_digest,
            "before_fingerprint": before_fingerprint,
            "expected_value": expected,
        }
    )
    record = TypedInputLineage(
        version=TYPED_INPUT_LINEAGE_VERSION,
        device_id=device_id,
        exact_value=expected,
        app_id=app_id,
        screen_id=screen_id,
        input_meaning="application_text_input",
        input_field_id=input_field_id,
        input_bounds=_valid_bounds(before_input["bounds"]),
        before_fingerprint=before_fingerprint,
        after_fingerprint="pending-visual-verification",
        action_digest=action_digest,
        receipt_digest=receipt_digest,
        surface_descriptors=(),
        recorded_at_epoch=(
            time.time() if recorded_at_epoch is None else float(recorded_at_epoch)
        ),
        source="pending_verified_text_action",
    )
    record.validate()
    return record


def _validated_newline_action_chain(
    resolved: Any,
    before_scene: Any,
) -> tuple[dict[str, Any], str, str]:
    if (
        not isinstance(resolved, dict)
        or resolved.get("kind") != "press_enter"
        or not isinstance(before_scene, dict)
    ):
        raise InputValueLineageError("换行连续性只接受已解析的 Enter 动作。")
    prior = resolved.get("prior_input_value")
    expected = resolved.get("expected_input_value")
    target_id = resolved.get("target_element_id")
    expected_effect = resolved.get("expected_effect")
    expected_state = (
        expected_effect.get("element_state")
        if isinstance(expected_effect, dict)
        else None
    )
    expected_states = (
        expected_state.get("states")
        if isinstance(expected_state, dict)
        else None
    )
    if (
        not isinstance(prior, str)
        or not isinstance(expected, str)
        or "\r" in prior
        or expected != prior + "\n"
        or not isinstance(target_id, str)
        or expected_state is None
        or expected_state.get("meaning") != "application_text_input"
        or expected_states != {"value": expected}
    ):
        raise InputValueLineageError("换行动作的 prior/newline/expected 链无效。")
    before_input = _single_input(before_scene, expected_value=prior)
    elements = before_scene.get("elements")
    targets = [
        item
        for item in elements or []
        if isinstance(item, dict) and item.get("element_id") == target_id
    ]
    if len(targets) != 1:
        raise InputValueLineageError("换行动作缺少唯一 Enter 目标。")
    target = targets[0]
    states = target.get("states")
    if (
        target.get("meaning") != "input_exact_enter_key"
        or not isinstance(states, dict)
        or states.get("input_enter_key") is not True
        or states.get("key_action") != "newline"
        or states.get("key_value") != "\n"
        or states.get("prior_input_value") != prior
        or states.get("expected_input_value") != expected
        or states.get("input_element_id") != before_input.get("element_id")
        or states.get("input_field_id") != _typed_input_field_id(before_input)
    ):
        raise InputValueLineageError("换行目标没有绑定同一 typed multiline 输入框。")
    return before_input, prior, expected


def build_pending_newline_lineage(
    *,
    device_id: str,
    resolved_action: dict[str, Any],
    before_scene: dict[str, Any],
    hardware_receipt: dict[str, Any],
    recorded_at_epoch: float | None = None,
) -> TypedInputLineage:
    """Bind one exact Enter event to its immediate post-action observation."""

    if (
        not isinstance(hardware_receipt, dict)
        or hardware_receipt.get("seller_event_barrier_confirmed") is not True
        or hardware_receipt.get("round_trip_position_confirmed") is not True
        or hardware_receipt.get("mechanical_contact_ack") is not False
    ):
        raise InputValueLineageError("临时换行连续性缺少有效事件栅栏。")
    before_input, _prior, expected = _validated_newline_action_chain(
        resolved_action,
        before_scene,
    )
    app_id = before_scene.get("app_id")
    screen_id = before_scene.get("screen_id")
    before_fingerprint = before_scene.get("fingerprint")
    if any(
        not isinstance(value, str) or not value.strip() or value == "unknown"
        for value in (app_id, screen_id, before_fingerprint)
    ):
        raise InputValueLineageError("临时换行连续性缺少明确输入表面。")
    record = TypedInputLineage(
        version=TYPED_INPUT_LINEAGE_VERSION,
        device_id=device_id,
        exact_value=expected,
        app_id=app_id,
        screen_id=screen_id,
        input_meaning="application_text_input",
        input_field_id=_typed_input_field_id(before_input),
        input_bounds=_valid_bounds(before_input["bounds"]),
        before_fingerprint=before_fingerprint,
        after_fingerprint="pending-visual-verification",
        action_digest=_canonical_digest(resolved_action),
        receipt_digest=_canonical_digest(hardware_receipt),
        surface_descriptors=(),
        recorded_at_epoch=(
            time.time() if recorded_at_epoch is None else float(recorded_at_epoch)
        ),
        source="pending_verified_newline_action",
    )
    record.validate()
    return record


def build_pending_input_state_lineage(
    *,
    device_id: str,
    resolved_action: dict[str, Any],
    before_scene: dict[str, Any],
    hardware_receipt: dict[str, Any],
    recorded_at_epoch: float | None = None,
) -> TypedInputLineage:
    """Bind a verified keyboard-state switch that must preserve exact text."""

    if (
        not isinstance(resolved_action, dict)
        or resolved_action.get("kind") != "tap_semantic"
        or not isinstance(hardware_receipt, dict)
        or hardware_receipt.get("seller_event_barrier_confirmed") is not True
        or hardware_receipt.get("round_trip_position_confirmed") is not True
        or hardware_receipt.get("mechanical_contact_ack") is not False
    ):
        raise InputValueLineageError(
            "临时输入状态连续性缺少有效单击事件栅栏。"
        )
    prior = resolved_action.get("prior_input_value")
    expected = resolved_action.get("expected_input_value")
    expected_effect = resolved_action.get("expected_effect")
    expected_state = (
        expected_effect.get("element_state")
        if isinstance(expected_effect, dict)
        else None
    )
    expected_states = (
        expected_state.get("states")
        if isinstance(expected_state, dict)
        else None
    )
    if (
        not isinstance(prior, str)
        or not prior
        or prior != expected
        or "\r" in prior
        or "\n" in prior
        or not isinstance(expected_state, dict)
        or expected_state.get("meaning") != "application_text_input"
        or not isinstance(expected_states, dict)
        or expected_states.get("value") != prior
        or len(expected_states) != 2
    ):
        raise InputValueLineageError(
            "临时输入状态连续性的同值 expected 合同无效。"
        )
    before_input = _single_input(before_scene, expected_value=prior)
    elements = before_scene.get("elements") if isinstance(before_scene, dict) else None
    targets = [
        item
        for item in elements or []
        if isinstance(item, dict)
        and item.get("element_id") == resolved_action.get("target_element_id")
    ]
    if len(targets) != 1:
        raise InputValueLineageError(
            "临时输入状态连续性缺少唯一输入辅助键。"
        )
    target = targets[0]
    states = target.get("states")
    meaning = target.get("meaning")
    if (
        not isinstance(states, dict)
        or states.get("prior_input_value") != prior
        or states.get("input_element_id") != before_input.get("element_id")
    ):
        raise InputValueLineageError(
            "临时输入状态连续性没有绑定原输入框和值。"
        )
    expected_state_key = {
        "switch_keyboard_layout": ("keyboard_layout", "target_layout"),
        "switch_keyboard_case": ("keyboard_case_mode", "target_mode"),
        "switch_keyboard_input_mode": ("keyboard_input_mode", "target_mode"),
    }.get(str(meaning or ""))
    if expected_state_key is None:
        raise InputValueLineageError(
            "临时输入状态连续性只接受键盘布局、大小写或输入模式切换。"
        )
    state_key, target_key = expected_state_key
    if expected_states.get(state_key) != states.get(target_key):
        raise InputValueLineageError(
            "临时输入状态连续性的切换方向与 expected 不一致。"
        )
    app_id = before_scene.get("app_id")
    screen_id = before_scene.get("screen_id")
    before_fingerprint = before_scene.get("fingerprint")
    if any(
        not isinstance(value, str) or not value.strip() or value == "unknown"
        for value in (app_id, screen_id, before_fingerprint)
    ):
        raise InputValueLineageError(
            "临时输入状态连续性缺少明确输入表面。"
        )
    record = TypedInputLineage(
        version=TYPED_INPUT_LINEAGE_VERSION,
        device_id=device_id,
        exact_value=prior,
        app_id=app_id,
        screen_id=screen_id,
        input_meaning="application_text_input",
        input_field_id=_typed_input_field_id(before_input),
        input_bounds=_valid_bounds(before_input["bounds"]),
        before_fingerprint=before_fingerprint,
        after_fingerprint="pending-visual-verification",
        action_digest=_canonical_digest(resolved_action),
        receipt_digest=_canonical_digest(hardware_receipt),
        surface_descriptors=(),
        recorded_at_epoch=(
            time.time() if recorded_at_epoch is None else float(recorded_at_epoch)
        ),
        source="pending_verified_input_state_action",
    )
    record.validate()
    return record


def build_pending_ime_candidate_lineage(
    *,
    device_id: str,
    resolved_action: dict[str, Any],
    before_scene: dict[str, Any],
    hardware_receipt: dict[str, Any],
    recorded_at_epoch: float | None = None,
) -> TypedInputLineage:
    """Bind one exact IME candidate commit to its typed application field."""

    if (
        not isinstance(resolved_action, dict)
        or resolved_action.get("kind") != "tap_semantic"
        or not isinstance(hardware_receipt, dict)
        or hardware_receipt.get("seller_event_barrier_confirmed") is not True
        or hardware_receipt.get("round_trip_position_confirmed") is not True
        or hardware_receipt.get("mechanical_contact_ack") is not False
    ):
        raise InputValueLineageError(
            "临时候选提交连续性缺少有效单击事件栅栏。"
        )
    prior = resolved_action.get("prior_input_value")
    expected = resolved_action.get("expected_input_value")
    target_id = resolved_action.get("target_element_id")
    expected_effect = resolved_action.get("expected_effect")
    expected_state = (
        expected_effect.get("element_state")
        if isinstance(expected_effect, dict)
        else None
    )
    expected_states = (
        expected_state.get("states")
        if isinstance(expected_state, dict)
        else None
    )
    if (
        not isinstance(prior, str)
        or not isinstance(expected, str)
        or not expected
        or expected == prior
        or not expected.startswith(prior)
        or "\r" in prior
        or "\n" in prior
        or "\r" in expected
        or "\n" in expected
        or not isinstance(target_id, str)
        or not target_id
        or not isinstance(expected_state, dict)
        or expected_state.get("meaning") != "application_text_input"
        or expected_states != {"value": expected}
        or not str(resolved_action.get("formal_candidate_id") or "").strip()
    ):
        raise InputValueLineageError(
            "临时候选提交连续性的 prior/expected 合同无效。"
        )
    before_input = _single_input(before_scene, expected_value=prior)
    input_states = before_input.get("states")
    field_id = (
        str(input_states.get("input_field_id") or "").strip()
        if isinstance(input_states, dict)
        else ""
    )
    if field_id in {"", "unknown"}:
        raise InputValueLineageError(
            "临时候选提交连续性缺少 typed input_field_id。"
        )
    elements = before_scene.get("elements")
    candidates = [
        item
        for item in elements or []
        if isinstance(item, dict)
        and item.get("role") == "button"
        and item.get("meaning") == "ime_exact_candidate"
        and item.get("label") == expected
        and isinstance(item.get("states"), dict)
        and item["states"].get("ime_candidate") is True
        and item["states"].get("input_element_id")
        == before_input.get("element_id")
        and item["states"].get("prior_input_value") == prior
        and item["states"].get("expected_input_value") == expected
        and item["states"].get("goal_relevant") is True
        and item["states"].get("fully_visible") is True
        and item["states"].get("independent_geometry_verified") is True
    ]
    if len(candidates) != 1 or candidates[0].get("element_id") != target_id:
        raise InputValueLineageError(
            "临时候选提交连续性缺少唯一绑定当前输入框的精确候选。"
        )
    preedit = input_states.get("ime_preedit_text")
    candidate_states = candidates[0]["states"]
    if (
        not isinstance(preedit, str)
        or not preedit
        or candidate_states.get("pinyin") != preedit
    ):
        raise InputValueLineageError(
            "临时候选提交连续性没有绑定同一输入法预编辑串。"
        )
    app_id = before_scene.get("app_id")
    screen_id = before_scene.get("screen_id")
    before_fingerprint = before_scene.get("fingerprint")
    if any(
        not isinstance(value, str) or not value.strip() or value == "unknown"
        for value in (app_id, screen_id, before_fingerprint)
    ):
        raise InputValueLineageError(
            "临时候选提交连续性缺少明确输入表面。"
        )
    record = TypedInputLineage(
        version=TYPED_INPUT_LINEAGE_VERSION,
        device_id=device_id,
        exact_value=expected,
        app_id=app_id,
        screen_id=screen_id,
        input_meaning="application_text_input",
        input_field_id=field_id,
        input_bounds=_valid_bounds(before_input["bounds"]),
        before_fingerprint=before_fingerprint,
        after_fingerprint="pending-visual-verification",
        action_digest=_canonical_digest(resolved_action),
        receipt_digest=_canonical_digest(hardware_receipt),
        surface_descriptors=(),
        recorded_at_epoch=(
            time.time() if recorded_at_epoch is None else float(recorded_at_epoch)
        ),
        source="pending_verified_ime_candidate_action",
    )
    record.validate()
    return record


def build_pending_literal_lineage(
    *,
    device_id: str,
    resolved_action: dict[str, Any],
    before_scene: dict[str, Any],
    hardware_receipt: dict[str, Any],
    recorded_at_epoch: float | None = None,
) -> TypedInputLineage:
    """Build a non-persistent expectation for the first post-action view."""

    if not isinstance(resolved_action, dict) or resolved_action.get("kind") != "tap_semantic":
        raise InputValueLineageError("临时输入连续性只接受逐字符点击。")
    if (
        not isinstance(hardware_receipt, dict)
        or hardware_receipt.get("seller_event_barrier_confirmed") is not True
        or hardware_receipt.get("round_trip_position_confirmed") is not True
        or hardware_receipt.get("mechanical_contact_ack") is not False
    ):
        raise InputValueLineageError("临时输入连续性缺少有效事件栅栏。")
    prior = resolved_action.get("prior_input_value")
    expected = resolved_action.get("expected_input_value")
    target_id = resolved_action.get("target_element_id")
    expected_effect = resolved_action.get("expected_effect")
    expected_state = expected_effect.get("element_state") if isinstance(expected_effect, dict) else None
    expected_states = expected_state.get("states") if isinstance(expected_state, dict) else None
    if (
        not isinstance(prior, str)
        or not isinstance(expected, str)
        or "\r" in prior
        or "\n" in prior
        or "\r" in expected
        or "\n" in expected
        or expected_state is None
        or expected_state.get("meaning") != "application_text_input"
        or expected_states != {"value": expected}
    ):
        raise InputValueLineageError("临时输入连续性的 expected 合同无效。")
    before_input = _single_input(before_scene, expected_value=prior)
    elements = before_scene.get("elements") if isinstance(before_scene, dict) else None
    keys = [
        item
        for item in elements or []
        if isinstance(item, dict) and item.get("element_id") == target_id
    ]
    if len(keys) != 1:
        raise InputValueLineageError("临时输入连续性缺少唯一目标键。")
    key = keys[0]
    states = key.get("states")
    key_value = states.get("key_value") if isinstance(states, dict) else None
    if (
        key.get("meaning") != "input_exact_literal_key"
        or not isinstance(key_value, str)
        or len(key_value) != 1
        or key_value in {"\r", "\n"}
        or expected != prior + key_value
        or states.get("prior_input_value") != prior
        or states.get("expected_input_value") != expected
        or states.get("input_element_id") != before_input.get("element_id")
        or states.get("independent_geometry_verified") is not True
    ):
        raise InputValueLineageError("临时输入连续性目标键未形成 exact 链。")
    app_id = before_scene.get("app_id")
    screen_id = before_scene.get("screen_id")
    before_fingerprint = before_scene.get("fingerprint")
    if any(
        not isinstance(value, str) or not value.strip() or value == "unknown"
        for value in (app_id, screen_id, before_fingerprint)
    ):
        raise InputValueLineageError("临时输入连续性缺少明确输入表面。")
    record = TypedInputLineage(
        version=TYPED_INPUT_LINEAGE_VERSION,
        device_id=device_id,
        exact_value=expected,
        app_id=app_id,
        screen_id=screen_id,
        input_meaning="application_text_input",
        input_field_id=_typed_input_field_id(before_input),
        input_bounds=_valid_bounds(before_input["bounds"]),
        before_fingerprint=before_fingerprint,
        after_fingerprint="pending-visual-verification",
        action_digest=_canonical_digest(resolved_action),
        receipt_digest=_canonical_digest(hardware_receipt),
        surface_descriptors=(),
        recorded_at_epoch=(
            time.time() if recorded_at_epoch is None else float(recorded_at_epoch)
        ),
        source="pending_verified_literal_action",
    )
    record.validate()
    return record


def _single_input(scene: dict[str, Any], *, expected_value: str | None = None) -> dict[str, Any]:
    elements = scene.get("elements") if isinstance(scene, dict) else None
    if not isinstance(elements, list):
        raise InputValueLineageError("场景缺少 elements。")
    candidates = []
    for element in elements:
        states = element.get("states") if isinstance(element, dict) else None
        if (
            isinstance(states, dict)
            and element.get("role") == "input"
            and element.get("meaning") == "application_text_input"
            and isinstance(element.get("confidence"), (int, float))
            and not isinstance(element.get("confidence"), bool)
            and float(element["confidence"]) >= 0.9
            and states.get("fully_visible") is True
            and states.get("focused") is True
            and isinstance(states.get("value"), str)
            and (expected_value is None or states.get("value") == expected_value)
            and _valid_bounds(element.get("bounds")) is not None
        ):
            candidates.append(element)
    if len(candidates) != 1:
        raise InputValueLineageError("场景没有唯一可信聚焦输入框。")
    return candidates[0]


def _typed_input_field_id(element: dict[str, Any]) -> str:
    states = element.get("states") if isinstance(element, dict) else None
    field_id = (
        str(states.get("input_field_id") or "").strip()
        if isinstance(states, dict)
        else ""
    )
    return field_id if field_id and field_id != "unknown" else "unknown"


def _validated_text_action_chain(
    resolved: Any,
    before_scene: Any,
) -> tuple[dict[str, Any], str, str, str]:
    if (
        not isinstance(resolved, dict)
        or resolved.get("kind") != "input_verified_text"
        or resolved.get("input_method") != "direct_latin"
        or not isinstance(before_scene, dict)
    ):
        raise InputValueLineageError("文字连续性只接受已解析的英文直输分段。")
    prior = resolved.get("prior_input_value")
    expected = resolved.get("expected_input_value")
    fragment = resolved.get("input_fragment")
    expected_effect = resolved.get("expected_effect")
    expected_state = (
        expected_effect.get("element_state")
        if isinstance(expected_effect, dict)
        else None
    )
    expected_states = (
        expected_state.get("states")
        if isinstance(expected_state, dict)
        else None
    )
    if (
        not isinstance(prior, str)
        or not isinstance(expected, str)
        or not isinstance(fragment, str)
        or not fragment
        or "\r" in prior
        or "\r" in expected
        or "\r" in fragment
        or "\n" in fragment
        or expected != prior + fragment
        or expected_state is None
        or expected_state.get("meaning") != "application_text_input"
        or expected_states != {"value": expected}
    ):
        raise InputValueLineageError("文字输入分段的 prior/fragment/expected 链无效。")
    return _single_input(before_scene, expected_value=prior), prior, expected, fragment


def _record_from_text_execution(
    *,
    device_id: str,
    resolved: Any,
    before_scene: Any,
    after_scene: Any,
    recorded_at_epoch: float,
    source: str,
    surface_fallback: TypedInputLineage | None = None,
    surface_frames: tuple[Image.Image, ...] | None = None,
) -> TypedInputLineage:
    if not isinstance(after_scene, dict):
        raise InputValueLineageError("文字连续性缺少动作后场景。")
    before_input, prior, expected, _fragment = _validated_text_action_chain(
        resolved,
        before_scene,
    )
    before_fingerprint = before_scene.get("fingerprint")
    after_fingerprint = after_scene.get("fingerprint")
    if (
        not isinstance(before_fingerprint, str)
        or not isinstance(after_fingerprint, str)
        or before_fingerprint == after_fingerprint
    ):
        raise InputValueLineageError("文字连续性缺少变化后的 fingerprint。")
    after_input = _single_input(after_scene)
    raw_after = after_input["states"]["value"]
    if (
        not _exact_or_soft_wrapped_visual_text(raw_after, expected)
        or not any(raw_after in str(item) for item in after_input.get("evidence", []))
        or not _bounds_compatible(
            _valid_bounds(before_input["bounds"]),
            _valid_bounds(after_input["bounds"]),
        )
    ):
        raise InputValueLineageError("动作后文字值或输入表面与 exact 分段不一致。")
    app_id = after_scene.get("app_id")
    screen_id = after_scene.get("screen_id")
    fallback_compatible = bool(
        surface_fallback is not None
        and surface_fallback.device_id == device_id
        and surface_fallback.exact_value == prior
        and _bounds_compatible(
            surface_fallback.input_bounds,
            _valid_bounds(before_input["bounds"]),
        )
        and _surface_identity_compatible(
            recorded_app_id=surface_fallback.app_id,
            recorded_screen_id=surface_fallback.screen_id,
            current_app_id=str(before_scene.get("app_id") or ""),
            current_screen_id=str(before_scene.get("screen_id") or ""),
            exact_value=prior,
        )
    )
    if fallback_compatible:
        app_id = surface_fallback.app_id
        screen_id = surface_fallback.screen_id
    if not isinstance(app_id, str) or not app_id.strip() or app_id == "unknown":
        raise InputValueLineageError("文字连续性缺少明确 app_id。")
    if not isinstance(screen_id, str) or not screen_id.strip() or screen_id == "unknown":
        raise InputValueLineageError("文字连续性缺少明确 screen_id。")
    action_digest = _canonical_digest(resolved)
    receipt_digest = _canonical_digest(
        {
            "protocol_version": "2026-08-20-verified-text-transaction-v1",
            "stage": "post_action_exact_verified",
            "device_id": device_id,
            "action_digest": action_digest,
            "before_fingerprint": before_fingerprint,
            "after_fingerprint": after_fingerprint,
            "expected_value": expected,
        }
    )
    record = TypedInputLineage(
        version=TYPED_INPUT_LINEAGE_VERSION,
        device_id=device_id,
        exact_value=expected,
        app_id=app_id,
        screen_id=screen_id,
        input_meaning="application_text_input",
        input_field_id=_typed_input_field_id(after_input),
        input_bounds=_valid_bounds(after_input["bounds"]),
        before_fingerprint=before_fingerprint,
        after_fingerprint=after_fingerprint,
        action_digest=action_digest,
        receipt_digest=receipt_digest,
        surface_descriptors=_surface_descriptors(
            surface_frames,
            _valid_bounds(after_input["bounds"]),
        ),
        recorded_at_epoch=recorded_at_epoch,
        source=source,
    )
    record.validate()
    return record


def _record_from_newline_execution(
    *,
    device_id: str,
    resolved: Any,
    before_scene: Any,
    after_scene: Any,
    hardware_receipt: Any,
    recorded_at_epoch: float,
    source: str,
    surface_fallback: TypedInputLineage | None = None,
    surface_frames: tuple[Image.Image, ...] | None = None,
) -> TypedInputLineage:
    if not isinstance(after_scene, dict):
        raise InputValueLineageError("换行连续性缺少动作后场景。")
    if (
        not isinstance(hardware_receipt, dict)
        or hardware_receipt.get("seller_event_barrier_confirmed") is not True
        or hardware_receipt.get("round_trip_position_confirmed") is not True
        or hardware_receipt.get("mechanical_contact_ack") is not False
    ):
        raise InputValueLineageError("换行连续性缺少有效单击事件栅栏。")
    before_input, prior, expected = _validated_newline_action_chain(
        resolved,
        before_scene,
    )
    before_fingerprint = before_scene.get("fingerprint")
    after_fingerprint = after_scene.get("fingerprint")
    if (
        not isinstance(before_fingerprint, str)
        or not isinstance(after_fingerprint, str)
        or before_fingerprint == after_fingerprint
    ):
        raise InputValueLineageError("换行连续性缺少变化后的 fingerprint。")
    after_input = _single_input(after_scene, expected_value=expected)
    after_states = after_input.get("states")
    after_evidence = after_input.get("evidence")
    if (
        not isinstance(after_states, dict)
        or after_states.get("value") != expected
        or after_states.get("input_field_id") != _typed_input_field_id(before_input)
        or after_states.get("verified_trailing_newline") is not True
        or not isinstance(after_evidence, list)
        or not any("已验证换行动作" in str(item) for item in after_evidence)
        or not _bounds_compatible(
            _valid_bounds(before_input["bounds"]),
            _valid_bounds(after_input["bounds"]),
        )
    ):
        raise InputValueLineageError("动作后场景没有证明同一字段的真实尾随换行。")
    app_id = after_scene.get("app_id")
    screen_id = after_scene.get("screen_id")
    fallback_compatible = bool(
        surface_fallback is not None
        and surface_fallback.device_id == device_id
        and surface_fallback.exact_value == prior
        and surface_fallback.input_field_id == _typed_input_field_id(before_input)
        and _bounds_compatible(
            surface_fallback.input_bounds,
            _valid_bounds(before_input["bounds"]),
        )
    )
    if fallback_compatible:
        app_id = surface_fallback.app_id
        screen_id = surface_fallback.screen_id
    if not isinstance(app_id, str) or not app_id.strip() or app_id == "unknown":
        raise InputValueLineageError("换行连续性缺少明确 app_id。")
    if not isinstance(screen_id, str) or not screen_id.strip() or screen_id == "unknown":
        raise InputValueLineageError("换行连续性缺少明确 screen_id。")
    record = TypedInputLineage(
        version=TYPED_INPUT_LINEAGE_VERSION,
        device_id=device_id,
        exact_value=expected,
        app_id=app_id,
        screen_id=screen_id,
        input_meaning="application_text_input",
        input_field_id=_typed_input_field_id(after_input),
        input_bounds=_valid_bounds(after_input["bounds"]),
        before_fingerprint=before_fingerprint,
        after_fingerprint=after_fingerprint,
        action_digest=_canonical_digest(resolved),
        receipt_digest=_canonical_digest(hardware_receipt),
        surface_descriptors=_surface_descriptors(
            surface_frames,
            _valid_bounds(after_input["bounds"]),
        ),
        recorded_at_epoch=recorded_at_epoch,
        source=source,
    )
    record.validate()
    return record


def _record_from_execution(
    *,
    device_id: str,
    resolved: Any,
    before_scene: Any,
    after_scene: Any,
    hardware_receipt: Any,
    recorded_at_epoch: float,
    source: str,
    surface_fallback: TypedInputLineage | None = None,
    surface_frames: tuple[Image.Image, ...] | None = None,
) -> TypedInputLineage:
    if not isinstance(resolved, dict) or resolved.get("kind") != "tap_semantic":
        raise InputValueLineageError("只有已验证的逐字符点击能形成输入值连续性。")
    if not isinstance(before_scene, dict) or not isinstance(after_scene, dict):
        raise InputValueLineageError("输入值连续性缺少前后场景。")
    if (
        not isinstance(hardware_receipt, dict)
        or hardware_receipt.get("seller_event_barrier_confirmed") is not True
        or hardware_receipt.get("round_trip_position_confirmed") is not True
        or hardware_receipt.get("mechanical_contact_ack") is not False
    ):
        raise InputValueLineageError("输入值连续性缺少有效的单击事件栅栏。")
    before_fingerprint = before_scene.get("fingerprint")
    after_fingerprint = after_scene.get("fingerprint")
    if not isinstance(before_fingerprint, str) or not isinstance(after_fingerprint, str):
        raise InputValueLineageError("输入值连续性缺少前后 fingerprint。")
    prior = resolved.get("prior_input_value")
    expected = resolved.get("expected_input_value")
    target_id = resolved.get("target_element_id")
    expected_effect = resolved.get("expected_effect")
    expected_state = expected_effect.get("element_state") if isinstance(expected_effect, dict) else None
    expected_states = expected_state.get("states") if isinstance(expected_state, dict) else None
    if (
        not isinstance(prior, str)
        or not isinstance(expected, str)
        or "\r" in prior
        or "\n" in prior
        or "\r" in expected
        or "\n" in expected
        or not isinstance(target_id, str)
        or expected_state is None
        or expected_state.get("meaning") != "application_text_input"
        or expected_states != {"value": expected}
    ):
        raise InputValueLineageError("逐字符动作的 prior/expected 合同无效。")
    before_input = _single_input(before_scene, expected_value=prior)
    elements = before_scene.get("elements")
    keys = [item for item in elements if isinstance(item, dict) and item.get("element_id") == target_id]
    if len(keys) != 1:
        raise InputValueLineageError("逐字符动作没有唯一目标键。")
    key = keys[0]
    key_states = key.get("states")
    key_value = key_states.get("key_value") if isinstance(key_states, dict) else None
    if (
        key.get("meaning") != "input_exact_literal_key"
        or not isinstance(key_value, str)
        or len(key_value) != 1
        or key_value in {"\r", "\n"}
        or expected != prior + key_value
        or key_states.get("prior_input_value") != prior
        or key_states.get("expected_input_value") != expected
        or key_states.get("input_element_id") != before_input.get("element_id")
        or key_states.get("independent_geometry_verified") is not True
    ):
        raise InputValueLineageError("逐字符目标键没有形成严格 exact 链。")
    after_input = _single_input(after_scene)
    raw_after = after_input["states"]["value"]
    if (
        _collapsed_visual_text(raw_after) != expected
        or not any(raw_after in str(item) for item in after_input.get("evidence", []))
        or not _bounds_compatible(
            _valid_bounds(before_input["bounds"]),
            _valid_bounds(after_input["bounds"]),
        )
    ):
        raise InputValueLineageError("动作后输入值或输入表面与 exact 回执不一致。")
    app_id = after_scene.get("app_id")
    screen_id = after_scene.get("screen_id")
    fallback_compatible = bool(
        surface_fallback is not None
        and surface_fallback.device_id == device_id
        and surface_fallback.exact_value == prior
        and _bounds_compatible(
            surface_fallback.input_bounds,
            _valid_bounds(before_input["bounds"]),
        )
        and _surface_identity_compatible(
            recorded_app_id=surface_fallback.app_id,
            recorded_screen_id=surface_fallback.screen_id,
            current_app_id=str(before_scene.get("app_id") or ""),
            current_screen_id=str(before_scene.get("screen_id") or ""),
            exact_value=prior,
        )
    )
    if fallback_compatible:
        app_id = surface_fallback.app_id
        screen_id = surface_fallback.screen_id
    if not isinstance(app_id, str) or not app_id.strip() or app_id == "unknown":
        raise InputValueLineageError("输入值连续性缺少明确 app_id。")
    if not isinstance(screen_id, str) or not screen_id.strip() or screen_id == "unknown":
        raise InputValueLineageError("输入值连续性缺少明确 screen_id。")
    record = TypedInputLineage(
        version=TYPED_INPUT_LINEAGE_VERSION,
        device_id=device_id,
        exact_value=expected,
        app_id=app_id,
        screen_id=screen_id,
        input_meaning="application_text_input",
        input_field_id=_typed_input_field_id(after_input),
        input_bounds=_valid_bounds(after_input["bounds"]),
        before_fingerprint=before_fingerprint,
        after_fingerprint=after_fingerprint,
        action_digest=_canonical_digest(resolved),
        receipt_digest=_canonical_digest(hardware_receipt),
        surface_descriptors=_surface_descriptors(
            surface_frames,
            _valid_bounds(after_input["bounds"]),
        ),
        recorded_at_epoch=recorded_at_epoch,
        source=source,
    )
    record.validate()
    return record
