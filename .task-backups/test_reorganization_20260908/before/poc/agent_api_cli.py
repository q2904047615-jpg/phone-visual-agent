"""Command-line entry point for the only supported local Agent API client."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Mapping

try:
    from .local_agent_api_client import LocalAgentApiClient, LocalAgentApiError
except ImportError:  # Direct execution: python poc/agent_api_cli.py ...
    from local_agent_api_client import LocalAgentApiClient, LocalAgentApiError


def _session_projection(response: Mapping[str, Any]) -> dict[str, Any]:
    session = response.get("session") if isinstance(response.get("session"), dict) else {}
    execution = response.get("execution") if isinstance(response.get("execution"), dict) else {}
    return {
        "ok": True,
        "mode": response.get("mode"),
        "request_physical_actions": response.get("physical_actions"),
        "session_id": session.get("session_id"),
        "device_id": session.get("device_id"),
        "status": session.get("status"),
        "revision": session.get("revision"),
        "total_physical_actions": session.get("physical_actions"),
        "execution_budget": session.get("execution_budget"),
        "auto_pause_reason": session.get("auto_pause_reason"),
        "confirmation_ready": session.get("confirmation_ready"),
        "confirmation_scope": session.get("confirmation_scope"),
        "failed_reason": session.get("failed_reason"),
        "action_outcome": execution.get("action_outcome"),
        "visual_outcome": execution.get("visual_outcome"),
        "verification_errors": execution.get("verification_errors"),
    }


def _device_projection(response: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "controller_online": response.get("controller_online"),
        "camera_online": response.get("camera_online"),
        "busy": response.get("busy"),
        "stop_requested": response.get("stop_requested"),
        "default_device_id": response.get("default_device_id"),
        "active_tasks": response.get("active_tasks"),
        "generic_supervised_execution": response.get("generic_supervised_execution"),
    }


def _doctor_projection(response: Mapping[str, Any]) -> dict[str, Any]:
    device = response.get("device") if isinstance(response.get("device"), dict) else {}
    controller = (
        response.get("controller")
        if isinstance(response.get("controller"), dict)
        else {}
    )
    camera = response.get("camera") if isinstance(response.get("camera"), dict) else {}
    providers = (
        response.get("providers")
        if isinstance(response.get("providers"), dict)
        else {}
    )
    qwen = providers.get("qwen") if isinstance(providers.get("qwen"), dict) else {}
    capabilities = (
        response.get("capabilities")
        if isinstance(response.get("capabilities"), dict)
        else {}
    )
    return {
        "ok": True,
        "ready": response.get("ready"),
        "physical_actions": response.get("physical_actions"),
        "device_id": device.get("device_id"),
        "exclusive_available": device.get("exclusive_available"),
        "controller_online": controller.get("controller_online"),
        "camera_online": controller.get("camera_online"),
        "camera_stable": (
            camera.get("stability", {}).get("stable")
            if isinstance(camera.get("stability"), dict)
            else None
        ),
        "qwen_configured": qwen.get("configured"),
        "qwen_model": qwen.get("model"),
        "supported_actions": capabilities.get("supported_actions"),
        "blockers": response.get("blockers"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="类型化本地 Agent API 网关")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--full", action="store_true", help="输出完整非令牌响应")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("bootstrap")
    subparsers.add_parser("status")
    doctor = subparsers.add_parser("doctor")
    doctor.add_argument("--device-id", default="device-local-01")
    start = subparsers.add_parser("start")
    start.add_argument("--device-id", default="device-local-01")
    start.add_argument("--text", required=True)
    start.add_argument("--exact-input-text")
    start.add_argument(
        "--exact-action-kind",
        choices=("back", "home", "open_recent_apps", "tap_semantic"),
    )
    start.add_argument("--exact-target-label", default="")
    start.add_argument("--auto-advance", action="store_true")
    start.add_argument("--max-physical-actions", type=int)
    start.add_argument("--max-observations", type=int)
    for name in ("get", "confirm-once", "next", "cancel", "pause", "auto"):
        command = subparsers.add_parser(name)
        command.add_argument("--session-id", required=True)
        if name == "auto":
            command.add_argument("--max-physical-actions", type=int)
            command.add_argument("--max-observations", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        with LocalAgentApiClient(
            base_url=args.base_url,
            timeout_seconds=args.timeout,
        ) as client:
            if args.command == "bootstrap":
                result = {"ok": True, **client.bootstrap()}
            elif args.command == "status":
                raw = client.device_status()
                result = raw if args.full else _device_projection(raw)
            elif args.command == "doctor":
                raw = client.doctor(args.device_id)
                result = raw if args.full else _doctor_projection(raw)
            elif args.command == "start":
                raw = client.start_session(
                    text=args.text,
                    exact_input_text=args.exact_input_text,
                    exact_action_kind=args.exact_action_kind,
                    exact_target_label=args.exact_target_label,
                    device_id=args.device_id,
                    auto_advance=args.auto_advance,
                    max_physical_actions=args.max_physical_actions,
                    max_observations=args.max_observations,
                )
                result = raw if args.full else _session_projection(raw)
            elif args.command == "get":
                raw = client.get_session(args.session_id)
                result = raw if args.full else _session_projection(raw)
            elif args.command == "confirm-once":
                raw = client.confirm_once(args.session_id)
                result = raw if args.full else _session_projection(raw)
            elif args.command == "next":
                raw = client.plan_next(args.session_id)
                result = raw if args.full else _session_projection(raw)
            elif args.command == "auto":
                raw = client.continue_automatic(args.session_id,
                    max_physical_actions=args.max_physical_actions, max_observations=args.max_observations)
                result = raw if args.full else _session_projection(raw)
            elif args.command == "cancel":
                raw = client.cancel_session(args.session_id)
                result = raw if args.full else _session_projection(raw)
            else:
                raw = client.pause_session(args.session_id)
                result = raw if args.full else _session_projection(raw)
    except LocalAgentApiError as exc:
        print(
            json.dumps(
                {"ok": False, "error": exc.details.to_dict()},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
