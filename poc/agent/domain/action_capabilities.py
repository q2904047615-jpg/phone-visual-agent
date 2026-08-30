from __future__ import annotations

from .validation import ValidatedDataclassWire, canonical_digest, reject_if
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


CAPABILITY_PROTOCOL = "2026-08-19-action-capability-v1"
CAPABILITY_GAP_PROTOCOL = "2026-08-19-capability-gap-v1"

PROMOTABLE_ACTIONS = frozenset({'tap_semantic', 'dismiss_overlay', 'swipe', 'back', 'home', 'reveal_system_navigation',
    'input_verified_text', 'double_tap', 'long_press', 'drag'})

CALIBRATION_BOUND_ACTIONS = frozenset({'double_tap', 'long_press', 'drag', 'reveal_system_navigation'})

KNOWN_ACTION_CAPABILITIES = frozenset({'tap_semantic', 'dismiss_overlay', 'swipe', 'reveal_system_navigation', 'back',
    'home', 'open_recent_apps', 'wait_for_change', 'input_verified_text', 'clear_verified_text', 'long_press', 'drag',
    'double_tap', 'press_enter', 'launch_app', 'pinch', 'hardware_key'})

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class ActionCapabilityError(ValueError):
    pass


@dataclass(frozen=True)
class CapabilityGap(ValidatedDataclassWire):
    device_id: str
    requested_action: str
    reason_code: str
    supported_actions: tuple[str, ...]
    required_parameters: tuple[str, ...] = ()
    profile_digest: str = ""
    protocol_version: str = CAPABILITY_GAP_PROTOCOL

    def validate(self) -> None:
        reject_if(self.protocol_version != CAPABILITY_GAP_PROTOCOL, ActionCapabilityError("CapabilityGap protocol_version 无效。"))
        reject_if(not _ID.fullmatch(self.device_id), ActionCapabilityError("CapabilityGap device_id 无效。"))
        reject_if(self.requested_action not in KNOWN_ACTION_CAPABILITIES, ActionCapabilityError("CapabilityGap requested_action 未知。"))
        reject_if(not _ID.fullmatch(self.reason_code), ActionCapabilityError("CapabilityGap reason_code 无效。"))
        reject_if(len(self.supported_actions) != len(set(self.supported_actions)), ActionCapabilityError("CapabilityGap supported_actions 重复。"))
        reject_if(set(self.supported_actions) - KNOWN_ACTION_CAPABILITIES, ActionCapabilityError("CapabilityGap 含未知 supported action。"))
        reject_if(self.profile_digest and (not re.fullmatch('[0-9a-f]{64}', self.profile_digest)), ActionCapabilityError("CapabilityGap profile_digest 无效。"))

@dataclass(frozen=True)
class DeviceCapabilitySnapshot:
    device_id: str
    actions: dict[str, dict[str, Any]]
    protocol_version: str = CAPABILITY_PROTOCOL

    def validate(self) -> None:
        reject_if(self.protocol_version != CAPABILITY_PROTOCOL, ActionCapabilityError("capability protocol_version 无效。"))
        reject_if(not _ID.fullmatch(self.device_id), ActionCapabilityError("capability device_id 无效。"))
        reject_if(set(self.actions) != KNOWN_ACTION_CAPABILITIES, ActionCapabilityError("capability actions 必须完整覆盖已知动作。"))
        for (action, spec) in self.actions.items():
            reject_if(not isinstance(spec, dict) or not isinstance(spec.get('enabled'), bool), ActionCapabilityError(f"capability.{action} 缺少 enabled。"))
            json.dumps(spec, ensure_ascii=False, allow_nan=False)

    @property
    def profile_digest(self) -> str:
        return canonical_digest(self.to_dict(include_digest=False))

    @property
    def supported_actions(self) -> tuple[str, ...]:
        return tuple(sorted(action for action, spec in self.actions.items() if spec["enabled"]))

    def to_dict(self, *, include_digest: bool=True) -> dict[str, Any]:
        self.validate()
        value = {'protocol_version': self.protocol_version, 'device_id': self.device_id,
            'actions': {key: dict(self.actions[key]) for key in sorted(self.actions)}}
        if include_digest:
            value["profile_digest"] = self.profile_digest
        return value

    def gap(self, requested_action: str, *, required_parameters: Iterable[str]=()) -> CapabilityGap | None:
        self.validate()
        reject_if(requested_action not in KNOWN_ACTION_CAPABILITIES, ActionCapabilityError(f"未知动作能力：{requested_action}"))
        if self.actions[requested_action]['enabled']:
            return None
        return CapabilityGap(device_id=self.device_id, requested_action=requested_action,
            reason_code=str(self.actions[requested_action].get('gap_reason') or 'device_capability_not_verified'),
            supported_actions=self.supported_actions, required_parameters=tuple((str(item) for item
            in required_parameters)), profile_digest=self.profile_digest)


def build_device_capability_snapshot(*, device_id: str, supported_actions: Iterable[str], raw_profile: Mapping[str,
    Any] | None=None) -> DeviceCapabilitySnapshot:
    supported = frozenset(str(item) for item in supported_actions)
    unknown = supported - KNOWN_ACTION_CAPABILITIES
    reject_if(unknown, ActionCapabilityError('设备声明未知动作能力：' + ', '.join(sorted(unknown))))
    raw_actions = raw_profile.get('actions', {}) if isinstance(raw_profile, Mapping) else {}
    if not isinstance(raw_actions, Mapping):
        raw_actions = {}
    default_gap = {'double_tap': 'requires_double_tap_live_acceptance',
        'press_enter': 'requires_fresh_visible_enter_key', 'pinch': 'multi_touch_not_supported_by_single_contact_robot',
        'launch_app': 'trusted_package_launch_not_configured', 'hardware_key': 'hardware_key_transport_not_verified'}
    actions: dict[str, dict[str, Any]] = {}
    for action in sorted(KNOWN_ACTION_CAPABILITIES):
        supplied = raw_actions.get(action)
        spec = dict(supplied) if isinstance(supplied, Mapping) else {}
        spec["enabled"] = action in supported
        if action not in supported:
            spec['gap_reason'] = str(spec.get('gap_reason') or default_gap.get(action)
                or 'device_capability_not_verified')
        actions[action] = spec
    snapshot = DeviceCapabilitySnapshot(device_id=str(device_id), actions=actions)
    snapshot.validate()
    return snapshot
