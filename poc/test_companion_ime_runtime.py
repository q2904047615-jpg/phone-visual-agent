from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from agent.domain.text_transport import TEXT_TRANSPORT_PROTOCOL
from agent.infrastructure import companion_ime_runtime as runtime_module
from agent.infrastructure.companion_ime_runtime import (
    COMPANION_IME_RUNTIME_REGISTRY_VERSION,
    CompanionImeDeviceRuntimeConfig,
    CompanionImeRuntimeConfigError,
    CompanionImeRuntimeRegistry,
)
from agent.infrastructure.windows_companion_pairing_store import (
    StoredCompanionPairing,
)


def _profile(device_id: str, *, enabled: bool = True) -> dict:
    return {
        "protocol_version": TEXT_TRANSPORT_PROTOCOL,
        "profile_id": f"profile-{device_id}",
        "device_id": device_id,
        "pairing_id": f"pairing-{device_id}",
        "enabled": enabled,
        "capabilities": ["append_text", "clear_text"],
        "ack_timeout_seconds": 5.0,
    }


def _device(
    device_id: str,
    *,
    port: int,
    enabled: bool = True,
) -> dict:
    return {
        "profile": _profile(device_id, enabled=enabled),
        "bind_host": "127.0.0.1",
        "bind_port": port,
        "tls_certificate_path": "tls/server.crt",
        "tls_private_key_path": "tls/server.key",
    }


class _FakeBridge:
    def __init__(self, events: list[str], device_id: str, *, fail_start: bool = False):
        self._events = events
        self._device_id = device_id
        self._fail_start = fail_start

    def start(self):
        self._events.append(f"start:{self._device_id}")
        if self._fail_start:
            raise OSError("occupied")
        return "127.0.0.1", 12345

    def stop(self):
        self._events.append(f"stop:{self._device_id}")


class _FakeAuthority:
    def __init__(self, events: list[str], device_id: str):
        self._events = events
        self._device_id = device_id

    def close(self):
        self._events.append(f"close:{self._device_id}")


class CompanionImeRuntimeRegistryTests(unittest.TestCase):
    @staticmethod
    def _write_registry(directory: Path, devices: list[dict]) -> Path:
        path = directory / "companion-ime.json"
        path.write_text(
            json.dumps({
                "version": COMPANION_IME_RUNTIME_REGISTRY_VERSION,
                "devices": devices,
            }),
            encoding="utf-8",
        )
        return path

    def test_missing_registry_keeps_all_devices_on_mechanical_transport(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            called = []
            registry = CompanionImeRuntimeRegistry(
                Path(temporary) / "missing.json",
                pairing_state_directory=Path(temporary) / "state",
                device_runtime_factory=lambda config, state: called.append((config, state)),
            )

            self.assertEqual((), registry.configured_device_ids)
            self.assertIsNone(registry.transport_for_device("device-local-01"))
            registry.start()
            self.assertTrue(registry.started)
            registry.stop()
            self.assertFalse(registry.started)
            self.assertEqual([], called)

    def test_only_enabled_device_gets_one_transport_and_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry_path = self._write_registry(root, [
                _device("device-a", port=18766),
                _device("device-disabled", port=18766, enabled=False),
            ])
            events: list[str] = []
            transports: dict[str, object] = {}
            states: list[Path] = []

            def factory(config, state_directory):
                device_id = config.profile.device_id
                states.append(state_directory)
                transport = transports.setdefault(device_id, object())
                return SimpleNamespace(
                    profile=config.profile,
                    transport=transport,
                    bridge=_FakeBridge(events, device_id),
                    pairing_authority=_FakeAuthority(events, device_id),
                )

            state_directory = root / "pairing-state"
            registry = CompanionImeRuntimeRegistry(
                registry_path,
                pairing_state_directory=state_directory,
                device_runtime_factory=factory,
            )

            self.assertEqual(("device-a",), registry.configured_device_ids)
            self.assertIs(transports["device-a"], registry.transport_for_device("device-a"))
            self.assertIsNone(registry.transport_for_device("device-disabled"))
            self.assertIsNone(registry.transport_for_device("other-device"))
            self.assertEqual([state_directory], states)
            registry.start()
            registry.start()
            registry.stop()
            self.assertEqual([
                "start:device-a",
                "stop:device-a",
                "close:device-a",
            ], events)

    def test_selected_device_builds_only_one_runtime_and_exposes_safe_setup_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tls_directory = root / "tls"
            tls_directory.mkdir()
            (tls_directory / "server.crt").write_text(
                "-----BEGIN CERTIFICATE-----\nYWJj\n-----END CERTIFICATE-----\n",
                encoding="ascii",
            )
            registry_path = self._write_registry(root, [
                _device("device-a", port=18766),
                _device("device-b", port=18767),
            ])
            built_devices: list[str] = []
            credential = StoredCompanionPairing(
                device_id="device-b",
                pairing_id="pairing-device-b",
                installation_id="installation-b",
                created_at_epoch=123.0,
                _shared_key=b"x" * 32,
            )

            def factory(config, _state_directory):
                device_id = config.profile.device_id
                built_devices.append(device_id)
                return SimpleNamespace(
                    config=config,
                    profile=config.profile,
                    transport=object(),
                    bridge=_FakeBridge([], device_id),
                    pairing_authority=_FakeAuthority([], device_id),
                    pairing_store=SimpleNamespace(
                        load=lambda requested: credential
                        if requested == "device-b"
                        else None
                    ),
                )

            registry = CompanionImeRuntimeRegistry(
                registry_path,
                pairing_state_directory=root / "state",
                device_runtime_factory=factory,
                selected_device_id="device-b",
            )

            self.assertEqual(["device-b"], built_devices)
            self.assertEqual(("device-b",), registry.configured_device_ids)
            metadata = registry.setup_metadata_for_device("device-b")
            self.assertIsNotNone(metadata)
            self.assertEqual(hashlib.sha256(b"abc").hexdigest(), metadata.certificate_sha256)
            self.assertIsNone(registry.setup_metadata_for_device("device-a"))
            status = registry.pairing_status_for_device("device-b")
            self.assertEqual("installation-b", status.installation_id)
            safe_status = status.to_dict()
            self.assertNotIn("shared_key", safe_status)
            self.assertNotIn("protected_shared_key", safe_status)

    def test_unknown_selected_device_fails_before_runtime_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry_path = self._write_registry(root, [
                _device("device-a", port=18766),
            ])
            calls = []
            with self.assertRaisesRegex(
                CompanionImeRuntimeConfigError,
                "未配置或未启用",
            ):
                CompanionImeRuntimeRegistry(
                    registry_path,
                    pairing_state_directory=root / "state",
                    device_runtime_factory=lambda config, state: calls.append(
                        (config, state)
                    ),
                    selected_device_id="missing-device",
                )
            self.assertEqual([], calls)

    def test_start_failure_rolls_back_already_started_bridges(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry_path = self._write_registry(root, [
                _device("device-a", port=18766),
                _device("device-b", port=18767),
            ])
            events: list[str] = []

            def factory(config, _state_directory):
                device_id = config.profile.device_id
                return SimpleNamespace(
                    profile=config.profile,
                    transport=object(),
                    bridge=_FakeBridge(
                        events,
                        device_id,
                        fail_start=device_id == "device-b",
                    ),
                    pairing_authority=_FakeAuthority(events, device_id),
                )

            registry = CompanionImeRuntimeRegistry(
                registry_path,
                pairing_state_directory=root / "state",
                device_runtime_factory=factory,
            )

            with self.assertRaisesRegex(
                CompanionImeRuntimeConfigError,
                "TLS bridge 启动失败",
            ):
                registry.start()
            self.assertFalse(registry.started)
            self.assertEqual([
                "start:device-a",
                "start:device-b",
                "stop:device-b",
                "close:device-b",
                "stop:device-a",
                "close:device-a",
            ], events)

    def test_registry_rejects_duplicates_and_secret_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate = self._write_registry(root, [
                _device("device-a", port=18766),
                _device("device-a", port=18767),
            ])
            with self.assertRaisesRegex(
                CompanionImeRuntimeConfigError,
                "device_id 重复",
            ):
                CompanionImeRuntimeRegistry(
                    duplicate,
                    pairing_state_directory=root / "state",
                    device_runtime_factory=lambda _config, _state: None,
                )

            unsafe = _device("device-b", port=18768)
            unsafe["shared_key"] = "must-not-be-configurable"
            self._write_registry(root, [unsafe])
            with self.assertRaisesRegex(
                CompanionImeRuntimeConfigError,
                "包含额外字段",
            ):
                CompanionImeRuntimeRegistry(
                    duplicate,
                    pairing_state_directory=root / "state",
                    device_runtime_factory=lambda _config, _state: None,
                )

    def test_relative_tls_paths_resolve_against_registry_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = CompanionImeDeviceRuntimeConfig.from_dict(
                _device("device-a", port=18766),
                base_directory=root,
            )
            self.assertEqual((root / "tls" / "server.crt").resolve(), config.tls_certificate_path)
            self.assertEqual((root / "tls" / "server.key").resolve(), config.tls_private_key_path)

    def test_tls_context_loads_only_the_configured_certificate_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = CompanionImeDeviceRuntimeConfig.from_dict(
                _device("device-a", port=18766),
                base_directory=root,
            )
            loaded: list[tuple[str, str]] = []
            context = SimpleNamespace(
                minimum_version=None,
                load_cert_chain=lambda *, certfile, keyfile: loaded.append(
                    (certfile, keyfile)
                ),
            )

            with patch.object(
                runtime_module.ssl,
                "SSLContext",
                return_value=context,
            ) as context_factory:
                built = runtime_module._build_tls_context(config)

            self.assertIs(context, built)
            context_factory.assert_called_once_with(
                runtime_module.ssl.PROTOCOL_TLS_SERVER
            )
            self.assertEqual(runtime_module.ssl.TLSVersion.TLSv1_2, context.minimum_version)
            self.assertEqual([(
                str(config.tls_certificate_path),
                str(config.tls_private_key_path),
            )], loaded)

    def test_production_builder_restores_pairing_before_server_construction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = CompanionImeDeviceRuntimeConfig.from_dict(
                _device("device-a", port=18766),
                base_directory=root,
            )
            events: list[str] = []
            pairing_tokens = object()
            one_time_tokens = object()
            store = object()
            bridge = object()
            transport = object()

            class Authority:
                def __init__(self, profile, one_time, credential_store, tokens):
                    self.profile = profile
                    self.one_time = one_time
                    self.store = credential_store
                    self.tokens = tokens

                def restore_persisted_pairing(self):
                    events.append("restore")

                def close(self):
                    events.append("close")

            def server_factory(profile, tokens, context, **kwargs):
                events.append("server")
                self.assertIs(config.profile, profile)
                self.assertIs(pairing_tokens, tokens)
                self.assertIs(kwargs["pairing_authority"], authority_instances[0])
                return bridge

            authority_instances: list[Authority] = []

            def authority_factory(*args):
                authority = Authority(*args)
                authority_instances.append(authority)
                return authority

            with (
                patch.object(runtime_module, "PairingTokenRegistry", return_value=pairing_tokens),
                patch.object(runtime_module, "OneTimePairingTokenRegistry", return_value=one_time_tokens),
                patch.object(runtime_module, "WindowsCompanionPairingStore", return_value=store),
                patch.object(runtime_module, "CompanionImePairingAuthority", side_effect=authority_factory),
                patch.object(runtime_module, "_build_tls_context", return_value=object()),
                patch.object(runtime_module, "TlsCompanionImeBridgeServer", side_effect=server_factory),
                patch.object(runtime_module, "CompanionImeTextTransport", return_value=transport),
            ):
                built = runtime_module._build_device_runtime(config, root / "state")

            self.assertEqual(["restore", "server"], events)
            self.assertIs(config, built.config)
            self.assertIs(transport, built.transport)
            self.assertIs(bridge, built.bridge)
            self.assertIs(store, built.pairing_store)


if __name__ == "__main__":
    unittest.main()
