"""Frame-bound one-shot orientation credentials for physical execution."""

from __future__ import annotations

import hashlib
import re
import threading
import uuid
import weakref
from dataclasses import dataclass, field
from typing import Any

from PIL import Image, ImageFilter


ORIENTATION_AUDIT_PROTOCOL_VERSION = "2026-08-15-orientation-audit-v2"
ORIENTATION_CREDENTIAL_VERSION = "2026-08-24-orientation-credential-v2"
MIN_ORIENTATION_CONFIDENCE = 0.80
MAX_ORIENTATION_MEAN_BRIGHTNESS_DELTA = 18.0
MAX_ORIENTATION_CENTERED_MAE = 6.0
ORIENTATION_AUDIT_SOURCE = "independent_orientation_audit"
LOCAL_QWERTY_ORIENTATION_SOURCE = "stable_local_qwerty_orientation_audit"
SINGLE_STEP_SCENE_ORIENTATION_SOURCE = "single_step_scene_orientation"
FIXED_SYSTEM_NAVIGATION_ACTIONS = frozenset(
    {"back", "home", "open_recent_apps"}
)
_PLACEHOLDER_DEVICE_IDS = frozenset({"", "unbound", "unknown", "none", "null"})
_AUDIT_SEAL_LOCK = threading.Lock()
_LIVE_AUDIT_SEALS: dict[object, "_FrameVisualBinding"] = {}
_CLAIMED_AUDIT_CREDENTIALS: weakref.WeakValueDictionary[object, Any] = (
    weakref.WeakValueDictionary()
)


class OrientationSafetyError(RuntimeError):
    pass


class OrientationFrameMismatchError(OrientationSafetyError):
    """Carry the exact rejected frame for session-local diagnostic evidence."""

    def __init__(
        self,
        message: str,
        *,
        actual_frame: Image.Image,
        brightness_delta: float,
        centered_mae: float,
    ) -> None:
        super().__init__(message)
        self.actual_frame = actual_frame.convert("RGB").copy()
        self.brightness_delta = float(brightness_delta)
        self.centered_mae = float(centered_mae)


def validate_device_id(value: str) -> str:
    device_id = str(value or "").strip()
    if device_id.casefold() in _PLACEHOLDER_DEVICE_IDS:
        raise OrientationSafetyError("方向安全必须绑定真实 device_id。")
    if len(device_id) > 128:
        raise OrientationSafetyError("device_id 过长。")
    return device_id


def camera_layout_orientation(size: tuple[int, int]) -> str:
    width, height = size
    if width > height:
        return "landscape"
    if height > width:
        return "portrait"
    return "square"


def frame_fingerprint(frame: Image.Image) -> str:
    compact = frame.convert("L").resize((64, 96), Image.Resampling.BILINEAR)
    return hashlib.sha256(compact.tobytes()).hexdigest()[:20]


@dataclass(frozen=True)
class _FrameVisualBinding:
    size: tuple[int, int]
    pixels: tuple[int, ...]
    mean: float


def _frame_visual_binding(frame: Image.Image) -> _FrameVisualBinding:
    compact = (
        frame.convert("L")
        .resize((64, 96), Image.Resampling.BILINEAR)
        .filter(ImageFilter.GaussianBlur(radius=0.8))
    )
    pixels = tuple(compact.tobytes())
    return _FrameVisualBinding(
        size=tuple(frame.size),
        pixels=pixels,
        mean=sum(pixels) / len(pixels),
    )


def _assert_visually_bound(
    reference: _FrameVisualBinding, actual_frame: Image.Image
) -> None:
    actual = _frame_visual_binding(actual_frame)
    if actual.size != reference.size:
        raise OrientationSafetyError("动作前实际捕获帧尺寸发生变化。")
    brightness_delta = abs(actual.mean - reference.mean)
    centered_mae = sum(
        abs((current - actual.mean) - (prior - reference.mean))
        for prior, current in zip(reference.pixels, actual.pixels)
    ) / len(reference.pixels)
    if (
        brightness_delta > MAX_ORIENTATION_MEAN_BRIGHTNESS_DELTA
        or centered_mae > MAX_ORIENTATION_CENTERED_MAE
    ):
        message = (
            "动作前实际捕获帧与独立方向审计帧发生视觉漂移："
            f"亮度差{brightness_delta:.2f}，结构差{centered_mae:.2f}。"
        )
        raise OrientationFrameMismatchError(
            message,
            actual_frame=actual_frame,
            brightness_delta=brightness_delta,
            centered_mae=centered_mae,
        )


@dataclass(frozen=True)
class OrientationCredential:
    version: str
    credential_id: str
    source: str
    device_id: str
    scene_fingerprint: str
    frame_fingerprint: str
    evidence_frame_fingerprint: str
    frame_size: tuple[int, int]
    camera_layout_orientation: str
    phone_content_rotation: str
    confidence: float
    evidence: tuple[str, ...]
    _audit_seal: object | None = field(
        default=None, repr=False, compare=False
    )

    def validate(self) -> None:
        if self.version != ORIENTATION_CREDENTIAL_VERSION:
            raise OrientationSafetyError("方向凭据版本无效。")
        if self.source not in {
            ORIENTATION_AUDIT_SOURCE,
            LOCAL_QWERTY_ORIENTATION_SOURCE,
            SINGLE_STEP_SCENE_ORIENTATION_SOURCE,
        }:
            raise OrientationSafetyError("方向凭据不是正式独立审计产生。")
        for label, value in (
            ("credential_id", self.credential_id),
            ("device_id", self.device_id),
            ("scene_fingerprint", self.scene_fingerprint),
            ("frame_fingerprint", self.frame_fingerprint),
        ):
            if not isinstance(value, str) or not value.strip() or len(value) > 128:
                raise OrientationSafetyError(f"方向凭据 {label} 无效。")
        if not isinstance(self.evidence_frame_fingerprint, str) or len(
            self.evidence_frame_fingerprint
        ) > 128:
            raise OrientationSafetyError("方向凭据持久化帧指纹无效。")
        if self.device_id.strip().casefold() in _PLACEHOLDER_DEVICE_IDS:
            raise OrientationSafetyError("方向凭据设备标识不能是占位值。")
        if (
            not isinstance(self.frame_size, tuple)
            or len(self.frame_size) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in self.frame_size)
        ):
            raise OrientationSafetyError("方向凭据画布尺寸无效。")
        local_layout = camera_layout_orientation(self.frame_size)
        if self.camera_layout_orientation != local_layout:
            raise OrientationSafetyError("方向凭据与本地画布尺寸冲突。")
        if self.phone_content_rotation not in {
            "upright", "rotated_90", "rotated_180", "rotated_270", "unknown"
        }:
            raise OrientationSafetyError("方向凭据手机内容方向无效。")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise OrientationSafetyError("方向凭据置信度无效。")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise OrientationSafetyError("方向凭据置信度超出范围。")
        if not isinstance(self.evidence, tuple) or not 1 <= len(self.evidence) <= 2:
            raise OrientationSafetyError("方向凭据必须包含一至两条只读证据。")
        forbidden = re.compile(
            r"(?:coordinates?|coords?|bounds?|\bx\s*[=:]|\by\s*[=:]|"
            r"\b(?:tap|click|press|swipe|drag|execute|suggest)\b|"
            r"点击|滑动|拖动|按下|坐标|执行|建议|机械臂|控制端|PX\s*/\s*MM)",
            re.IGNORECASE,
        )
        for item in self.evidence:
            if not isinstance(item, str) or not item.strip() or len(item) > 160:
                raise OrientationSafetyError("方向凭据只读证据无效。")
            if forbidden.search(item):
                raise OrientationSafetyError("方向凭据包含坐标、动作或外部控制端证据。")

    def assert_authorizes(
        self,
        *,
        device_id: str,
        scene_fingerprint: str,
        frame_size: tuple[int, int],
        action: str | None = None,
    ) -> None:
        self.validate()
        if self.device_id != device_id:
            raise OrientationSafetyError("方向凭据与设备不匹配。")
        if self.scene_fingerprint != scene_fingerprint:
            raise OrientationSafetyError("方向凭据与稳定场景不匹配。")
        if self.frame_size != tuple(frame_size):
            raise OrientationSafetyError("方向凭据与本地画布尺寸不匹配。")
        # These calibrated system-navigation primitives stay bound to the
        # exact device, scene, frame, confidence and one-shot live seal, but
        # do not depend on the central App content disclosing its rotation.
        # Visual/geometry actions remain strict, including callers that omit
        # an action scope.
        if (
            self.phone_content_rotation != "upright"
            and action not in FIXED_SYSTEM_NAVIGATION_ACTIONS
        ):
            raise OrientationSafetyError("手机内容方向不一致或未知。")
        if float(self.confidence) < MIN_ORIENTATION_CONFIDENCE:
            raise OrientationSafetyError("方向独立审计置信度不足。")

    def claim_live_execution_source(self) -> None:
        """Atomically claim this exact in-process credential for one promotion."""

        self.validate()
        seal = self._audit_seal
        with _AUDIT_SEAL_LOCK:
            if seal is None or _CLAIMED_AUDIT_CREDENTIALS.get(seal) is not self:
                raise OrientationSafetyError(
                    "方向凭据不是本进程已进入物理执行门的 live 对象。"
                )
            del _CLAIMED_AUDIT_CREDENTIALS[seal]

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "version": self.version,
            "credential_id": self.credential_id,
            "source": self.source,
            "device_id": self.device_id,
            "scene_fingerprint": self.scene_fingerprint,
            "frame_fingerprint": self.frame_fingerprint,
            "evidence_frame_fingerprint": self.evidence_frame_fingerprint,
            "frame_size": list(self.frame_size),
            "camera_layout_orientation": self.camera_layout_orientation,
            "phone_content_rotation": self.phone_content_rotation,
            "confidence": float(self.confidence),
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "OrientationCredential":
        if not isinstance(value, dict):
            raise OrientationSafetyError("方向凭据必须是对象。")
        required = {
            "version", "credential_id", "source", "device_id",
            "scene_fingerprint", "frame_fingerprint",
            "evidence_frame_fingerprint", "frame_size",
            "camera_layout_orientation", "phone_content_rotation",
            "confidence", "evidence",
        }
        if set(value) != required:
            raise OrientationSafetyError("方向凭据字段缺失或包含协议外字段。")
        size = value["frame_size"]
        evidence = value["evidence"]
        if not isinstance(size, list) or not isinstance(evidence, list):
            raise OrientationSafetyError("方向凭据尺寸或证据格式无效。")
        item = cls(
            version=value["version"], credential_id=value["credential_id"],
            source=value["source"], device_id=value["device_id"],
            scene_fingerprint=value["scene_fingerprint"],
            frame_fingerprint=value["frame_fingerprint"],
            evidence_frame_fingerprint=value["evidence_frame_fingerprint"],
            frame_size=tuple(size), camera_layout_orientation=value["camera_layout_orientation"],
            phone_content_rotation=value["phone_content_rotation"],
            confidence=value["confidence"], evidence=tuple(evidence),
        )
        item.validate()
        return item



def _mint_locally_verified_qwerty_credential(
    *,
    device_id: str,
    scene_fingerprint: str,
    frame: Image.Image,
    anchors: dict[str, Any],
) -> OrientationCredential:
    """Mint one action credential from stable local QWERTY row evidence.

    The caller supplies anchors returned by the production multi-frame OCR
    row snapper.  This function independently validates the complete upright
    row ordering before binding the resulting one-shot seal to the actual
    frame consumed by the physical execution gate.
    """

    required = ("q", "p", "a", "l", "z", "m", "backspace")
    if not isinstance(anchors, dict) or set(anchors) != set(required):
        raise OrientationSafetyError("本地方向审计缺少完整 QWERTY 七点。")

    points: dict[str, tuple[float, float]] = {}
    for key in required:
        value = anchors.get(key)
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(
                isinstance(part, bool)
                or not isinstance(part, (int, float))
                or not 0 <= float(part) <= 1000
                for part in value
            )
        ):
            raise OrientationSafetyError("本地方向审计的 QWERTY 锚点无效。")
        points[key] = (float(value[0]), float(value[1]))

    top_y = (points["q"][1] + points["p"][1]) / 2.0
    middle_y = (points["a"][1] + points["l"][1]) / 2.0
    bottom_y = (
        points["z"][1] + points["m"][1] + points["backspace"][1]
    ) / 3.0
    if not (
        points["q"][0] < points["p"][0]
        and points["a"][0] < points["l"][0]
        and points["z"][0] < points["m"][0] < points["backspace"][0]
        and top_y + 20 <= middle_y
        and middle_y + 20 <= bottom_y
        and abs(points["q"][1] - points["p"][1]) <= 18
        and abs(points["a"][1] - points["l"][1]) <= 18
        and max(
            points["z"][1],
            points["m"][1],
            points["backspace"][1],
        )
        - min(
            points["z"][1],
            points["m"][1],
            points["backspace"][1],
        )
        <= 18
    ):
        raise OrientationSafetyError("本地方向审计的 QWERTY 行序或水平结构无效。")

    seal = object()
    item = OrientationCredential(
        version=ORIENTATION_CREDENTIAL_VERSION,
        credential_id=uuid.uuid4().hex,
        source=LOCAL_QWERTY_ORIENTATION_SOURCE,
        device_id=validate_device_id(device_id),
        scene_fingerprint=str(scene_fingerprint or "").strip(),
        frame_fingerprint=frame_fingerprint(frame),
        evidence_frame_fingerprint="",
        frame_size=tuple(frame.size),
        camera_layout_orientation=camera_layout_orientation(frame.size),
        phone_content_rotation="upright",
        confidence=1.0,
        evidence=("本地连续多帧OCR确认完整QWERTY三行保持正向排列",),
        _audit_seal=seal,
    )
    item.validate()
    with _AUDIT_SEAL_LOCK:
        _LIVE_AUDIT_SEALS[seal] = _frame_visual_binding(frame)
    return item


def _mint_single_step_scene_credential(
    *,
    device_id: str,
    scene_fingerprint: str,
    frame: Image.Image,
    camera_layout_orientation_value: str,
    phone_content_rotation: str,
    confidence: float,
    evidence: tuple[str, ...],
) -> OrientationCredential:
    """Bind the sole step observation's direction facts to fresh local pixels.

    The model fact is already part of the parsed, fingerprint-bound UIScene.
    This function performs no model call.  It only validates that fact against
    the actual frame size and mints the same one-shot live seal used by the
    physical execution gate.
    """

    local_layout = camera_layout_orientation(tuple(frame.size))
    if camera_layout_orientation_value != local_layout:
        raise OrientationSafetyError(
            "单步画面方向与本地稳定帧尺寸不一致。"
        )
    seal = object()
    item = OrientationCredential(
        version=ORIENTATION_CREDENTIAL_VERSION,
        credential_id=uuid.uuid4().hex,
        source=SINGLE_STEP_SCENE_ORIENTATION_SOURCE,
        device_id=validate_device_id(device_id),
        scene_fingerprint=str(scene_fingerprint or "").strip(),
        frame_fingerprint=frame_fingerprint(frame),
        evidence_frame_fingerprint="",
        frame_size=tuple(frame.size),
        camera_layout_orientation=local_layout,
        phone_content_rotation=str(phone_content_rotation or "").strip(),
        confidence=float(confidence),
        evidence=tuple(evidence),
        _audit_seal=seal,
    )
    item.validate()
    if float(item.confidence) >= MIN_ORIENTATION_CONFIDENCE:
        with _AUDIT_SEAL_LOCK:
            _LIVE_AUDIT_SEALS[seal] = _frame_visual_binding(frame)
    return item


def _claim_audit_seal(credential: OrientationCredential) -> _FrameVisualBinding:
    seal = credential._audit_seal
    with _AUDIT_SEAL_LOCK:
        if seal is None or seal not in _LIVE_AUDIT_SEALS:
            raise OrientationSafetyError(
                "方向凭据不是本进程实际独立审计直接签发，或已使用。"
            )
        visual_binding = _LIVE_AUDIT_SEALS.pop(seal)
        _CLAIMED_AUDIT_CREDENTIALS[seal] = credential
        return visual_binding


class PhysicalExecutionGate:
    """One-shot direction authorization consumed before a physical primitive."""

    def __init__(self, device_id: str) -> None:
        self.device_id = validate_device_id(device_id)
        self._lock = threading.Lock()
        self._armed: tuple[
            str, OrientationCredential, _FrameVisualBinding
        ] | None = None

    def arm(self, credential: OrientationCredential, *, action: str, scene_fingerprint: str) -> None:
        with self._lock:
            # Every arm attempt replaces the authorization state atomically.
            # A malformed, stale, or already-consumed credential must never
            # leave an earlier authorization available to a later consume.
            self._armed = None
            credential.assert_authorizes(
                device_id=self.device_id,
                scene_fingerprint=scene_fingerprint,
                frame_size=credential.frame_size,
                action=action,
            )
            visual_binding = _claim_audit_seal(credential)
            self._armed = (
                action,
                credential,
                visual_binding,
            )

    def clear(self) -> None:
        with self._lock:
            self._armed = None

    def consume(self, *, action: str, frame: Image.Image) -> OrientationCredential:
        with self._lock:
            armed = self._armed
            self._armed = None
        if armed is None:
            raise OrientationSafetyError("物理执行缺少一次性方向授权。")
        armed_action, credential, visual_binding = armed
        if armed_action != action:
            raise OrientationSafetyError("一次性方向授权与物理动作不匹配。")
        credential.assert_authorizes(
            device_id=self.device_id,
            scene_fingerprint=credential.scene_fingerprint,
            frame_size=tuple(frame.size),
            action=action,
        )
        _assert_visually_bound(visual_binding, frame)
        return credential
