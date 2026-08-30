from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from companion_ime_setup import CompanionImeSetupError, main, run_setup
from agent.infrastructure.companion_ime_transport import CompletedCompanionImePairing
from agent.infrastructure.companion_ime_runtime import (
    CompanionImeSetupMetadata,
)


class _Grant:
    def __init__(self, token: str = "one-time-secret") -> None:
        self.expires_at_epoch = 12345.0
        self._token = token
        self.reveal_count = 0

    def reveal_token(self) -> str:
        self.reveal_count += 1
        return self._token


class _Authority:
    def __init__(self, grant: _Grant, statuses) -> None:
        self.grant = grant
        self.ttls: list[float] = []
        self.statuses = iter(statuses)

    def issue_one_time_token(self, *, ttl_seconds: float):
        self.ttls.append(ttl_seconds)
        return self.grant

    def completion_status(self):
        return next(self.statuses, None)


class _Registry:
    def __init__(
        self,
        statuses,
        *,
        bind_host: str = "192.0.2.10",
    ) -> None:
        self.metadata = CompanionImeSetupMetadata(
            device_id="device-a",
            profile_id="profile-device-a",
            pairing_id="pairing-device-a",
            bind_host=bind_host,
            bind_port=18766,
            certificate_sha256="a" * 64,
        )
        self.grant = _Grant()
        self.authority = _Authority(self.grant, statuses)
        self.events: list[str] = []

    def setup_metadata_for_device(self, device_id: str):
        self.events.append(f"metadata:{device_id}")
        return self.metadata

    def pairing_authority_for_device(self, device_id: str):
        self.events.append(f"authority:{device_id}")
        return self.authority

    def start(self):
        self.events.append("start")

    def stop(self):
        self.events.append("stop")


class CompanionImeSetupCliTests(unittest.TestCase):
    def test_success_waits_for_phone_promoted_signed_commit_not_just_persistence(self) -> None:
        previous = CompletedCompanionImePairing(
            device_id="device-a",
            pairing_id="pairing-device-a",
            installation_id="installation-old",
            completed_at_epoch=11000.0,
        )
        paired = CompletedCompanionImePairing(
            device_id="device-a",
            pairing_id="pairing-device-a",
            installation_id="installation-a",
            completed_at_epoch=12000.0,
        )
        registry = _Registry([previous, previous, paired])
        factory_calls = []

        def factory(path, **kwargs):
            factory_calls.append((path, kwargs))
            return registry

        output = io.StringIO()
        errors = io.StringIO()
        sleeps: list[float] = []
        result = run_setup(
            device_id="device-a",
            registry_path=Path("registry.json"),
            pairing_state_directory=Path("pairing-state"),
            advertise_host=None,
            timeout_seconds=30.0,
            token_ttl_seconds=30.0,
            poll_interval_seconds=0.25,
            stdout=output,
            stderr=errors,
            registry_factory=factory,
            monotonic=lambda: 0.0,
            sleep=sleeps.append,
        )

        self.assertEqual(0, result)
        self.assertEqual("", errors.getvalue())
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(["pairing_ready", "pairing_succeeded"], [
            record["event"] for record in records
        ])
        self.assertEqual("one-time-secret", records[0]["one_time_token"])
        self.assertNotIn("one_time_token", records[1])
        self.assertNotIn("shared_key", output.getvalue())
        self.assertEqual(1, output.getvalue().count("one-time-secret"))
        self.assertEqual(1, registry.grant.reveal_count)
        self.assertEqual([30.0], registry.authority.ttls)
        self.assertEqual([0.25], sleeps)
        self.assertEqual("start", registry.events[2])
        self.assertEqual("stop", registry.events[-1])
        self.assertEqual("device-a", factory_calls[0][1]["selected_device_id"])

    def test_timeout_is_explicit_and_never_reprints_token(self) -> None:
        registry = _Registry([None, None, None])
        output = io.StringIO()
        errors = io.StringIO()
        clock = iter((0.0, 0.0, 1.0))
        result = run_setup(
            device_id="device-a",
            registry_path=Path("registry.json"),
            pairing_state_directory=Path("pairing-state"),
            advertise_host=None,
            timeout_seconds=1.0,
            token_ttl_seconds=10.0,
            poll_interval_seconds=1.0,
            stdout=output,
            stderr=errors,
            registry_factory=lambda *_args, **_kwargs: registry,
            monotonic=lambda: next(clock),
            sleep=lambda _seconds: None,
        )

        self.assertEqual(1, result)
        self.assertEqual(1, output.getvalue().count("one-time-secret"))
        self.assertNotIn("shared_key", output.getvalue() + errors.getvalue())
        self.assertEqual(
            "pairing_timeout",
            json.loads(errors.getvalue())["event"],
        )
        self.assertEqual("stop", registry.events[-1])

    def test_wildcard_bind_requires_an_explicit_phone_reachable_host(self) -> None:
        registry = _Registry([], bind_host="0.0.0.0")
        with self.assertRaisesRegex(
            CompanionImeSetupError,
            "--advertise-host",
        ):
            run_setup(
                device_id="device-a",
                registry_path=Path("registry.json"),
                pairing_state_directory=Path("pairing-state"),
                advertise_host=None,
                timeout_seconds=30.0,
                token_ttl_seconds=30.0,
                poll_interval_seconds=0.25,
                registry_factory=lambda *_args, **_kwargs: registry,
            )
        self.assertEqual(0, registry.grant.reveal_count)
        self.assertNotIn("start", registry.events)

    def test_token_lifetime_cannot_expire_before_wait_timeout(self) -> None:
        with self.assertRaisesRegex(
            CompanionImeSetupError,
            "不能短于",
        ):
            run_setup(
                device_id="device-a",
                registry_path=Path("registry.json"),
                pairing_state_directory=Path("pairing-state"),
                advertise_host=None,
                timeout_seconds=60.0,
                token_ttl_seconds=30.0,
                poll_interval_seconds=0.25,
                registry_factory=lambda *_args, **_kwargs: SimpleNamespace(),
            )

    def test_main_reports_missing_device_as_contract_error_without_token(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            errors = io.StringIO()
            with redirect_stderr(errors):
                result = main([
                    "--device-id",
                    "missing-device",
                    "--registry",
                    str(Path(temporary) / "missing.json"),
                ])

        self.assertEqual(2, result)
        payload = json.loads(errors.getvalue())
        self.assertEqual("pairing_error", payload["event"])
        self.assertNotIn("one_time_token", errors.getvalue())
        self.assertNotIn("shared_key", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
