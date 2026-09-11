"""Frame-bound one-shot orientation credentials for physical execution."""

from __future__ import annotations

from agent.domain.validation import NormalizedPoint, dataclass_wire, reject_if
import re
import threading
import uuid
import weakref
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from agent.infrastructure.observation_images import local_frame_fingerprint as frame_fingerprint


ORIENTATION_CREDENTIAL_VERSION = "2026-08-24-orientation-credential-v2"
MIN_ORIENTATION_CONFIDENCE = 0.80
ORIENTATION_AUDIT_SOURCE = "independent_orientation_audit"
SINGLE_STEP_SCENE_ORIENTATION_SOURCE = "single_step_scene_orientation"
FIXED_SYSTEM_NAVIGATION_ACTIONS = frozenset({'back', 'home', 'open_recent_apps'})
_PLACEHOLDER_DEVICE_IDS = frozenset({"", "unbound", "unknown", "none", "null"})
_AUDIT_SEAL_LOCK = threading.Lock()
_LIVE_AUDIT_SEALS: dict[object, tuple[int, int]] = {}
_CLAIMED_AUDIT_CREDENTIALS: weakref.WeakValueDictionary[object, Any] = weakref.WeakValueDictionary()


class OrientationSafetyError(RuntimeError):
    pass


def validate_device_id(value: str) -> str:
    device_id = str(value or "").strip()
    reject_if(device_id.casefold() in _PLACEHOLDER_DEVICE_IDS, OrientationSafetyError("方向安全必须绑定真实 device_id。"))
    reject_if(len(device_id) > 128, OrientationSafetyError("device_id 过长。"))
    return device_id


def camera_layout_orientation(size: tuple[int, int]) -> str:
    width, height = size
    if width > height:
        return "landscape"
    if height > width:
        return "portrait"
    return "square"


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
    _audit_seal: object | None = field(default=None, repr=False, compare=False)

    def validate(self) -> None:
        reject_if(self.version != ORIENTATION_CREDENTIAL_VERSION, OrientationSafetyError("方向凭据版本无效。"))
        reject_if(
            self.source not in {ORIENTATION_AUDIT_SOURCE,
            SINGLE_STEP_SCENE_ORIENTATION_SOURCE},
            OrientationSafetyError("方向凭据不是正式独立审计产生。"),
        )
        for (label, value) in (('credential_id', self.credential_id), ('device_id', self.device_id),
            ('scene_fingerprint', self.scene_fingerprint), ('frame_fingerprint', self.frame_fingerprint)):
            reject_if(not isinstance(value, str) or not value.strip() or len(value) > 128, OrientationSafetyError(f"方向凭据 {label} 无效。"))
        reject_if(not isinstance(self.evidence_frame_fingerprint, str) or len(self.evidence_frame_fingerprint) > 128, OrientationSafetyError("方向凭据持久化帧指纹无效。"))
        reject_if(self.device_id.strip().casefold() in _PLACEHOLDER_DEVICE_IDS, OrientationSafetyError("方向凭据设备标识不能是占位值。"))
        reject_if(
            not isinstance(self.frame_size, tuple) or len(self.frame_size) != 2 or any((isinstance(v,
            bool) or not isinstance(v, int) or v <= 0 for v in self.frame_size)),
            OrientationSafetyError("方向凭据画布尺寸无效。"),
        )
        local_layout = camera_layout_orientation(self.frame_size)
        reject_if(self.camera_layout_orientation != local_layout, OrientationSafetyError("方向凭据与本地画布尺寸冲突。"))
        reject_if(self.phone_content_rotation not in {'upright', 'rotated_90', 'rotated_180', 'rotated_270', 'unknown'}, OrientationSafetyError("方向凭据手机内容方向无效。"))
        reject_if(isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)), OrientationSafetyError("方向凭据置信度无效。"))
        reject_if(not 0.0 <= float(self.confidence) <= 1.0, OrientationSafetyError("方向凭据置信度超出范围。"))
        reject_if(not isinstance(self.evidence, tuple) or not 1 <= len(self.evidence) <= 2, OrientationSafetyError("方向凭据必须包含一至两条只读证据。"))
        forbidden = re.compile(
            r"(?:coordinates?|coords?|bounds?|\bx\s*[=:]|\by\s*[=:]|"
            r"\b(?:tap|click|press|swipe|drag|execute|suggest)\b|"
            r"点击|滑动|拖动|按下|坐标|执行|建议|机械臂|控制端|PX\s*/\s*MM)",
            re.IGNORECASE,
        )
        for item in self.evidence:
            reject_if(not isinstance(item, str) or not item.strip() or len(item) > 160, OrientationSafetyError("方向凭据只读证据无效。"))
            reject_if(forbidden.search(item), OrientationSafetyError("方向凭据包含坐标、动作或外部控制端证据。"))

    def assert_authorizes(self, *, device_id: str, scene_fingerprint: str, frame_size: tuple[int, int],
        action: str | None=None) -> None:
        self.validate()
        reject_if(self.device_id != device_id, OrientationSafetyError("方向凭据与设备不匹配。"))
        reject_if(self.scene_fingerprint != scene_fingerprint, OrientationSafetyError("方向凭据与稳定场景不匹配。"))
        reject_if(self.frame_size != tuple(frame_size), OrientationSafetyError("方向凭据与本地画布尺寸不匹配。"))
        # A typed, explicit rotated value remains a physical-safety fact.  An
        # omitted/unknown model diagnostic does not become a second veto: the
        # locally minted single-step credential is already bound to the fresh
        # frame layout, device, scene, exact action and one-shot live seal.
        reject_if(self.phone_content_rotation in {'rotated_90', 'rotated_180', 'rotated_270'}
            and action not in FIXED_SYSTEM_NAVIGATION_ACTIONS, OrientationSafetyError("手机内容方向与执行坐标轴不一致。"))
        # Only the genuinely independent orientation-audit source owns an
        # authorization confidence.  Model confidence copied into a scene is
        # diagnostic and is never consulted by the locally minted source.
        reject_if(self.source == ORIENTATION_AUDIT_SOURCE and float(self.confidence) <
            MIN_ORIENTATION_CONFIDENCE, OrientationSafetyError("方向独立审计置信度不足。"))

    def claim_live_execution_source(self) -> None:
        """Atomically claim this exact in-process credential for one promotion."""

        self.validate()
        seal = self._audit_seal
        with _AUDIT_SEAL_LOCK:
            reject_if(seal is None or _CLAIMED_AUDIT_CREDENTIALS.get(seal) is not self, OrientationSafetyError('方向凭据不是本进程已进入物理执行门的 live 对象。'))
            del _CLAIMED_AUDIT_CREDENTIALS[seal]

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return dataclass_wire(self, omit=('_audit_seal',))

    @classmethod
    def from_dict(cls, value: Any) -> 'OrientationCredential':
        reject_if(not isinstance(value, dict), OrientationSafetyError("方向凭据必须是对象。"))
        required = {'version', 'credential_id', 'source', 'device_id', 'scene_fingerprint', 'frame_fingerprint',
            'evidence_frame_fingerprint', 'frame_size', 'camera_layout_orientation', 'phone_content_rotation',
            'confidence', 'evidence'}
        reject_if(set(value) != required, OrientationSafetyError("方向凭据字段缺失或包含协议外字段。"))
        size = value["frame_size"]
        evidence = value["evidence"]
        reject_if(not isinstance(size, list) or not isinstance(evidence, list), OrientationSafetyError("方向凭据尺寸或证据格式无效。"))
        item = cls(version=value['version'], credential_id=value['credential_id'], source=value['source'],
            device_id=value['device_id'], scene_fingerprint=value['scene_fingerprint'],
            frame_fingerprint=value['frame_fingerprint'], evidence_frame_fingerprint=value[
            'evidence_frame_fingerprint'], frame_size=tuple(size),
            camera_layout_orientation=value['camera_layout_orientation'],
            phone_content_rotation=value['phone_content_rotation'], confidence=value['confidence'],
            evidence=tuple(evidence))
        item.validate()
        return item




def _mint_single_step_scene_credential(*, device_id: str, scene_fingerprint: str,
    frame: Image.Image) -> OrientationCredential:
    """Mint a one-shot credential only from the fresh local frame binding.

    Qwen ``camera_alignment`` fields are optional diagnostics.  They are not
    accepted here, so they cannot become a second hardware authority.
    """

    local_layout = camera_layout_orientation(tuple(frame.size))
    seal = object()
    item = OrientationCredential(version=ORIENTATION_CREDENTIAL_VERSION, credential_id=uuid.uuid4().hex,
        source=SINGLE_STEP_SCENE_ORIENTATION_SOURCE, device_id=validate_device_id(device_id),
        scene_fingerprint=str(scene_fingerprint or '').strip(), frame_fingerprint=frame_fingerprint(frame),
        evidence_frame_fingerprint='', frame_size=tuple(frame.size), camera_layout_orientation=local_layout,
        phone_content_rotation='unknown', confidence=1.0,
        evidence=('本地当前帧尺寸与像素形成一次性方向绑定',), _audit_seal=seal)
    item.validate()
    with _AUDIT_SEAL_LOCK:
        _LIVE_AUDIT_SEALS[seal] = tuple(frame.size)
    return item


def _claim_audit_seal(credential: OrientationCredential) -> tuple[int, int]:
    seal = credential._audit_seal
    with _AUDIT_SEAL_LOCK:
        reject_if(seal is None or seal not in _LIVE_AUDIT_SEALS, OrientationSafetyError('方向凭据不是本进程实际独立审计直接签发，或已使用。'))
        frame_size = _LIVE_AUDIT_SEALS.pop(seal)
        _CLAIMED_AUDIT_CREDENTIALS[seal] = credential
        return frame_size


class PhysicalExecutionGate:
    """One-shot direction authorization consumed before a physical primitive."""

    def __init__(self, device_id: str) -> None:
        self.device_id = validate_device_id(device_id)
        self._lock = threading.Lock()
        self._armed: tuple[str, OrientationCredential, tuple[int, int]] | None = None

    def arm(self, credential: OrientationCredential, *, action: str, scene_fingerprint: str) -> None:
        with self._lock:
            # Every arm attempt replaces the authorization state atomically.
            # A malformed, stale, or already-consumed credential must never
            # leave an earlier authorization available to a later consume.
            self._armed = None
            credential.assert_authorizes(device_id=self.device_id, scene_fingerprint=scene_fingerprint,
                frame_size=credential.frame_size, action=action)
            frame_size = _claim_audit_seal(credential)
            self._armed = (action, credential, frame_size)

    def clear(self) -> None:
        with self._lock:
            self._armed = None

    def consume(self, *, action: str, frame: Image.Image) -> OrientationCredential:
        with self._lock:
            armed = self._armed
            self._armed = None
        reject_if(armed is None, OrientationSafetyError("物理执行缺少一次性方向授权。"))
        armed_action, credential, minted_frame_size = armed
        reject_if(armed_action != action, OrientationSafetyError("一次性方向授权与物理动作不匹配。"))
        credential.assert_authorizes(device_id=self.device_id, scene_fingerprint=credential.scene_fingerprint,
            frame_size=tuple(frame.size), action=action)
        # Pixel motion is not a second UI authority. The live seal still binds
        # the original canvas, including against a modified credential copy.
        reject_if(tuple(frame.size) != minted_frame_size,
            OrientationSafetyError("动作前实际捕获帧尺寸发生变化。"))
        return credential
