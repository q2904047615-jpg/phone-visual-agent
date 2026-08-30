"""Per-device Companion IME configuration and project-API lifecycle wiring."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import ssl
from typing import Any, Callable, Mapping

from agent.domain.text_transport import TextTransportProfile
from agent.infrastructure.companion_ime_transport import (
    CompanionImePairingAuthority,
    CompanionImeTextTransport,
    OneTimePairingTokenRegistry,
    PairingTokenRegistry,
    TlsCompanionImeBridgeServer,
)
from agent.infrastructure.windows_companion_pairing_store import (
    StoredCompanionPairing,
    WindowsCompanionPairingStore,
)


COMPANION_IME_RUNTIME_REGISTRY_VERSION = "2026-08-30-companion-ime-runtime-v1"
_REGISTRY_FIELDS = frozenset({"version", "devices"})
_DEVICE_FIELDS = frozenset({
    "profile",
    "bind_host",
    "bind_port",
    "tls_certificate_path",
    "tls_private_key_path",
})


class CompanionImeRuntimeConfigError(RuntimeError):
    """The local, secret-free Companion IME runtime registry is invalid."""


@dataclass(frozen=True)
class CompanionImeSetupMetadata:
    device_id: str
    profile_id: str
    pairing_id: str
    bind_host: str
    bind_port: int
    certificate_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "profile_id": self.profile_id,
            "pairing_id": self.pairing_id,
            "bind_host": self.bind_host,
            "bind_port": self.bind_port,
            "certificate_sha256": self.certificate_sha256,
        }


@dataclass(frozen=True)
class CompanionImePairingStatus:
    device_id: str
    pairing_id: str
    installation_id: str
    created_at_epoch: float

    @classmethod
    def from_credential(
        cls,
        credential: StoredCompanionPairing,
    ) -> "CompanionImePairingStatus":
        credential.validate()
        return cls(
            device_id=credential.device_id,
            pairing_id=credential.pairing_id,
            installation_id=credential.installation_id,
            created_at_epoch=float(credential.created_at_epoch),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "pairing_id": self.pairing_id,
            "installation_id": self.installation_id,
            "created_at_epoch": self.created_at_epoch,
        }


@dataclass(frozen=True)
class CompanionImeDeviceRuntimeConfig:
    profile: TextTransportProfile
    bind_host: str
    bind_port: int
    tls_certificate_path: Path
    tls_private_key_path: Path

    @classmethod
    def from_dict(
        cls,
        value: Any,
        *,
        base_directory: Path,
    ) -> "CompanionImeDeviceRuntimeConfig":
        if not isinstance(value, Mapping) or set(value) != _DEVICE_FIELDS:
            raise CompanionImeRuntimeConfigError(
                "Companion IME 设备配置字段不完整或包含额外字段。"
            )
        try:
            profile = TextTransportProfile.from_dict(value["profile"])
        except Exception as exc:
            raise CompanionImeRuntimeConfigError(
                "Companion IME profile 无效。"
            ) from exc
        host = value["bind_host"]
        port = value["bind_port"]
        if (
            not isinstance(host, str)
            or not host.strip()
            or "\x00" in host
            or len(host) > 255
        ):
            raise CompanionImeRuntimeConfigError(
                "Companion IME bind_host 无效。"
            )
        if (
            isinstance(port, bool)
            or not isinstance(port, int)
            or not 1 <= port <= 65535
        ):
            raise CompanionImeRuntimeConfigError(
                "Companion IME bind_port 无效。"
            )
        certificate = _resolve_file_path(
            value["tls_certificate_path"],
            base_directory=base_directory,
            label="tls_certificate_path",
        )
        private_key = _resolve_file_path(
            value["tls_private_key_path"],
            base_directory=base_directory,
            label="tls_private_key_path",
        )
        return cls(
            profile=profile,
            bind_host=host.strip(),
            bind_port=port,
            tls_certificate_path=certificate,
            tls_private_key_path=private_key,
        )


def _resolve_file_path(value: Any, *, base_directory: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise CompanionImeRuntimeConfigError(f"Companion IME {label} 无效。")
    path = Path(value.strip())
    if not path.is_absolute():
        path = base_directory / path
    return path.resolve()


def _certificate_sha256(path: Path) -> str:
    try:
        content = Path(path).read_bytes()
    except OSError as exc:
        raise CompanionImeRuntimeConfigError(
            "Companion IME TLS 证书不可读。"
        ) from exc
    match = re.search(
        rb"-----BEGIN CERTIFICATE-----\s+.+?\s+-----END CERTIFICATE-----",
        content,
        flags=re.DOTALL,
    )
    if match is None:
        raise CompanionImeRuntimeConfigError(
            "Companion IME TLS 证书不是有效 PEM 证书。"
        )
    try:
        pem = match.group(0).decode("ascii")
        der = ssl.PEM_cert_to_DER_cert(pem)
    except (UnicodeDecodeError, ValueError) as exc:
        raise CompanionImeRuntimeConfigError(
            "Companion IME TLS 证书无法计算 SHA-256 指纹。"
        ) from exc
    return hashlib.sha256(der).hexdigest()


@dataclass
class _CompanionImeDeviceRuntime:
    config: CompanionImeDeviceRuntimeConfig
    profile: TextTransportProfile
    transport: CompanionImeTextTransport
    bridge: TlsCompanionImeBridgeServer
    pairing_authority: CompanionImePairingAuthority
    pairing_store: WindowsCompanionPairingStore


DeviceRuntimeFactory = Callable[
    [CompanionImeDeviceRuntimeConfig, Path],
    _CompanionImeDeviceRuntime,
]


def _build_tls_context(config: CompanionImeDeviceRuntimeConfig) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    try:
        context.load_cert_chain(
            certfile=str(config.tls_certificate_path),
            keyfile=str(config.tls_private_key_path),
        )
    except (OSError, ssl.SSLError) as exc:
        raise CompanionImeRuntimeConfigError(
            f"Companion IME 设备 {config.profile.device_id!r} 的 TLS 证书不可用。"
        ) from exc
    return context


def _build_device_runtime(
    config: CompanionImeDeviceRuntimeConfig,
    pairing_state_directory: Path,
) -> _CompanionImeDeviceRuntime:
    pairing_tokens = PairingTokenRegistry()
    one_time_tokens = OneTimePairingTokenRegistry()
    pairing_store = WindowsCompanionPairingStore(pairing_state_directory)
    authority = CompanionImePairingAuthority(
        config.profile,
        one_time_tokens,
        pairing_store,
        pairing_tokens,
    )
    try:
        authority.restore_persisted_pairing()
        bridge = TlsCompanionImeBridgeServer(
            config.profile,
            pairing_tokens,
            _build_tls_context(config),
            host=config.bind_host,
            port=config.bind_port,
            pairing_authority=authority,
        )
        transport = CompanionImeTextTransport(
            config.profile,
            bridge,
            pairing_tokens,
        )
    except Exception:
        authority.close()
        raise
    return _CompanionImeDeviceRuntime(
        config=config,
        profile=config.profile,
        transport=transport,
        bridge=bridge,
        pairing_authority=authority,
        pairing_store=pairing_store,
    )


def _load_device_configs(registry_path: Path) -> tuple[CompanionImeDeviceRuntimeConfig, ...]:
    if not registry_path.exists():
        return ()
    try:
        raw = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompanionImeRuntimeConfigError(
            "Companion IME 运行配置不可读。"
        ) from exc
    if not isinstance(raw, Mapping) or set(raw) != _REGISTRY_FIELDS:
        raise CompanionImeRuntimeConfigError(
            "Companion IME 运行配置字段不完整或包含额外字段。"
        )
    if raw["version"] != COMPANION_IME_RUNTIME_REGISTRY_VERSION:
        raise CompanionImeRuntimeConfigError(
            "Companion IME 运行配置版本不匹配。"
        )
    devices = raw["devices"]
    if not isinstance(devices, list):
        raise CompanionImeRuntimeConfigError(
            "Companion IME devices 必须是数组。"
        )
    configs = tuple(
        CompanionImeDeviceRuntimeConfig.from_dict(
            item,
            base_directory=registry_path.parent,
        )
        for item in devices
    )
    _validate_unique_configs(configs)
    return configs


def _validate_unique_configs(
    configs: tuple[CompanionImeDeviceRuntimeConfig, ...],
) -> None:
    seen_devices: set[str] = set()
    seen_profiles: set[str] = set()
    seen_pairings: set[str] = set()
    seen_bindings: set[tuple[str, int]] = set()
    for config in configs:
        profile = config.profile
        if profile.device_id in seen_devices:
            raise CompanionImeRuntimeConfigError(
                f"Companion IME device_id 重复：{profile.device_id!r}。"
            )
        if profile.profile_id in seen_profiles:
            raise CompanionImeRuntimeConfigError(
                f"Companion IME profile_id 重复：{profile.profile_id!r}。"
            )
        if profile.pairing_id in seen_pairings:
            raise CompanionImeRuntimeConfigError(
                f"Companion IME pairing_id 重复：{profile.pairing_id!r}。"
            )
        seen_devices.add(profile.device_id)
        seen_profiles.add(profile.profile_id)
        seen_pairings.add(profile.pairing_id)
        if profile.enabled:
            binding = (config.bind_host, config.bind_port)
            if binding in seen_bindings:
                raise CompanionImeRuntimeConfigError(
                    "多个 Companion IME 设备不能监听同一 host/port。"
                )
            seen_bindings.add(binding)


class CompanionImeRuntimeRegistry:
    """Own exactly one optional Companion text transport per configured device."""

    def __init__(
        self,
        registry_path: Path,
        *,
        pairing_state_directory: Path,
        device_runtime_factory: DeviceRuntimeFactory | None = None,
        selected_device_id: str | None = None,
    ) -> None:
        self._registry_path = Path(registry_path)
        self._pairing_state_directory = Path(pairing_state_directory)
        factory = device_runtime_factory or _build_device_runtime
        self._devices: dict[str, _CompanionImeDeviceRuntime] = {}
        selected = str(selected_device_id or "").strip() or None
        try:
            configs = _load_device_configs(self._registry_path)
            if selected is not None and not any(
                config.profile.enabled and config.profile.device_id == selected
                for config in configs
            ):
                raise CompanionImeRuntimeConfigError(
                    f"Companion IME 设备 {selected!r} 未配置或未启用。"
                )
            for config in configs:
                if not config.profile.enabled:
                    continue
                if selected is not None and config.profile.device_id != selected:
                    continue
                runtime = factory(config, self._pairing_state_directory)
                if runtime.profile != config.profile:
                    raise CompanionImeRuntimeConfigError(
                        "Companion IME runtime profile 与设备配置不一致。"
                    )
                self._devices[config.profile.device_id] = runtime
        except Exception:
            for runtime in reversed(tuple(self._devices.values())):
                try:
                    runtime.bridge.stop()
                finally:
                    runtime.pairing_authority.close()
            raise
        self._started = False

    @property
    def configured_device_ids(self) -> tuple[str, ...]:
        return tuple(self._devices)

    @property
    def started(self) -> bool:
        return self._started

    def transport_for_device(
        self,
        device_id: str,
    ) -> CompanionImeTextTransport | None:
        runtime = self._devices.get(str(device_id or "").strip())
        return runtime.transport if runtime is not None else None

    def pairing_authority_for_device(
        self,
        device_id: str,
    ) -> CompanionImePairingAuthority | None:
        """Internal setup access; no HTTP route exposes this authority or its token."""

        runtime = self._devices.get(str(device_id or "").strip())
        return runtime.pairing_authority if runtime is not None else None

    def setup_metadata_for_device(
        self,
        device_id: str,
    ) -> CompanionImeSetupMetadata | None:
        """Return only non-secret local setup values for one configured device."""

        runtime = self._devices.get(str(device_id or "").strip())
        if runtime is None:
            return None
        config = runtime.config
        return CompanionImeSetupMetadata(
            device_id=config.profile.device_id,
            profile_id=config.profile.profile_id,
            pairing_id=config.profile.pairing_id,
            bind_host=config.bind_host,
            bind_port=config.bind_port,
            certificate_sha256=_certificate_sha256(
                config.tls_certificate_path
            ),
        )

    def pairing_status_for_device(
        self,
        device_id: str,
    ) -> CompanionImePairingStatus | None:
        """Read persisted safe metadata; never return protected or plaintext keys."""

        runtime = self._devices.get(str(device_id or "").strip())
        if runtime is None:
            return None
        credential = runtime.pairing_store.load(runtime.profile.device_id)
        if credential is None:
            return None
        return CompanionImePairingStatus.from_credential(credential)

    def start(self) -> None:
        if self._started:
            return
        try:
            for runtime in self._devices.values():
                runtime.bridge.start()
        except Exception as exc:
            for runtime in reversed(tuple(self._devices.values())):
                try:
                    runtime.bridge.stop()
                except Exception:
                    pass
                finally:
                    runtime.pairing_authority.close()
            raise CompanionImeRuntimeConfigError(
                "Companion IME TLS bridge 启动失败。"
            ) from exc
        self._started = True

    def stop(self) -> None:
        first_error: Exception | None = None
        for runtime in reversed(tuple(self._devices.values())):
            try:
                runtime.bridge.stop()
            except Exception as exc:
                first_error = first_error or exc
            finally:
                runtime.pairing_authority.close()
        self._started = False
        if first_error is not None:
            raise CompanionImeRuntimeConfigError(
                "Companion IME TLS bridge 停止失败。"
            ) from first_error


__all__ = [
    "COMPANION_IME_RUNTIME_REGISTRY_VERSION",
    "CompanionImeDeviceRuntimeConfig",
    "CompanionImePairingStatus",
    "CompanionImeRuntimeConfigError",
    "CompanionImeRuntimeRegistry",
    "CompanionImeSetupMetadata",
]
