"""Standalone one-time setup CLI for the project-owned Companion IME."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Callable, TextIO

from agent.domain.text_transport import TextTransportContractError
from agent.infrastructure.companion_ime_runtime import (
    CompanionImeRuntimeConfigError,
    CompanionImeRuntimeRegistry,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY_PATH = Path(os.environ.get(
    "ROBOT_COMPANION_IME_REGISTRY",
    ROOT / "companion_ime_registry.json",
))
DEFAULT_PAIRING_STATE_DIRECTORY = (
    ROOT / "output" / "web" / "state" / "companion_ime_pairings"
)


class CompanionImeSetupError(RuntimeError):
    """The local setup command cannot expose a usable one-time pairing grant."""


def _write_json(stream: TextIO, payload: dict) -> None:
    print(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        file=stream,
        flush=True,
    )


def _advertised_host(bind_host: str, override: str | None) -> str:
    host = str(override or "").strip()
    if host:
        if "\x00" in host or len(host) > 255:
            raise CompanionImeSetupError("--advertise-host 无效。")
        return host
    if bind_host in {"0.0.0.0", "::", "[::]"}:
        raise CompanionImeSetupError(
            "bind_host 是通配地址；请用 --advertise-host 指定手机可访问的电脑地址。"
        )
    return bind_host


def _validate_timing(
    *,
    timeout_seconds: float,
    token_ttl_seconds: float,
    poll_interval_seconds: float,
) -> None:
    if not 1.0 <= timeout_seconds <= 600.0:
        raise CompanionImeSetupError("--timeout-seconds 必须在 1 到 600 之间。")
    if not 10.0 <= token_ttl_seconds <= 600.0:
        raise CompanionImeSetupError("--token-ttl-seconds 必须在 10 到 600 之间。")
    if token_ttl_seconds < timeout_seconds:
        raise CompanionImeSetupError(
            "一次性 token 有效期不能短于配对等待超时。"
        )
    if not 0.05 <= poll_interval_seconds <= 5.0:
        raise CompanionImeSetupError("--poll-interval-seconds 必须在 0.05 到 5 之间。")


def run_setup(
    *,
    device_id: str,
    registry_path: Path,
    pairing_state_directory: Path,
    advertise_host: str | None,
    timeout_seconds: float,
    token_ttl_seconds: float,
    poll_interval_seconds: float,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    registry_factory: Callable[..., CompanionImeRuntimeRegistry] = (
        CompanionImeRuntimeRegistry
    ),
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Expose one token once, then wait for the phone's signed active-key commit."""

    selected_device = str(device_id or "").strip()
    if not selected_device:
        raise CompanionImeSetupError("--device-id 不能为空。")
    output = stdout or sys.stdout
    errors = stderr or sys.stderr
    _validate_timing(
        timeout_seconds=float(timeout_seconds),
        token_ttl_seconds=float(token_ttl_seconds),
        poll_interval_seconds=float(poll_interval_seconds),
    )
    registry = registry_factory(
        Path(registry_path),
        pairing_state_directory=Path(pairing_state_directory),
        selected_device_id=selected_device,
    )
    metadata = registry.setup_metadata_for_device(selected_device)
    authority = registry.pairing_authority_for_device(selected_device)
    if metadata is None or authority is None:
        raise CompanionImeSetupError(
            f"Companion IME 设备 {selected_device!r} 未配置或未启用。"
        )
    host = _advertised_host(metadata.bind_host, advertise_host)
    started = False
    try:
        registry.start()
        started = True
        before = authority.completion_status()
        grant = authority.issue_one_time_token(
            ttl_seconds=float(token_ttl_seconds)
        )
        _write_json(output, {
            "event": "pairing_ready",
            "device_id": selected_device,
            "host": host,
            "port": metadata.bind_port,
            "certificate_sha256": metadata.certificate_sha256,
            "one_time_token": grant.reveal_token(),
            "expires_at_epoch": grant.expires_at_epoch,
        })
        deadline = monotonic() + float(timeout_seconds)
        while True:
            current = authority.completion_status()
            if current is not None and current != before:
                _write_json(output, {
                    "event": "pairing_succeeded",
                    **current.to_dict(),
                })
                return 0
            remaining = deadline - monotonic()
            if remaining <= 0:
                _write_json(errors, {
                    "event": "pairing_timeout",
                    "device_id": selected_device,
                    "timeout_seconds": float(timeout_seconds),
                })
                return 1
            sleep(min(float(poll_interval_seconds), remaining))
    finally:
        if started:
            registry.stop()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "在项目 API 未运行时，为一个 Companion IME 设备生成一次性本地配对信息"
        )
    )
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument(
        "--advertise-host",
        help="手机可访问的电脑地址；bind_host 为 0.0.0.0/:: 时必填",
    )
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--token-ttl-seconds", type=float, default=300.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return run_setup(
            device_id=args.device_id,
            registry_path=args.registry,
            pairing_state_directory=DEFAULT_PAIRING_STATE_DIRECTORY,
            advertise_host=args.advertise_host,
            timeout_seconds=args.timeout_seconds,
            token_ttl_seconds=args.token_ttl_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        )
    except KeyboardInterrupt:
        _write_json(sys.stderr, {
            "event": "pairing_interrupted",
            "device_id": args.device_id,
        })
        return 130
    except (
        CompanionImeRuntimeConfigError,
        CompanionImeSetupError,
        TextTransportContractError,
        OSError,
    ) as exc:
        _write_json(sys.stderr, {
            "event": "pairing_error",
            "device_id": args.device_id,
            "error": str(exc),
        })
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
