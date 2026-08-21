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
    graph = session.get("task_graph") if isinstance(session.get("task_graph"), dict) else {}
    execution = response.get("execution") if isinstance(response.get("execution"), dict) else {}
    return {
        "ok": True,
        "mode": response.get("mode"),
        "request_physical_actions": response.get("physical_actions"),
        "session_id": session.get("session_id"),
        "device_id": session.get("device_id"),
        "status": session.get("status"),
        "revision": graph.get("revision"),
        "active_subgoal": graph.get("active_subgoal"),
        "total_physical_actions": session.get("physical_actions"),
        "confirmation_ready": session.get("confirmation_ready"),
        "confirmation_scope": session.get("confirmation_scope"),
        "failed_reason": session.get("failed_reason"),
        "action_outcome": execution.get("action_outcome"),
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="类型化本地 Agent API 网关")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--full", action="store_true", help="输出完整非令牌响应")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("bootstrap")
    subparsers.add_parser("status")
    start = subparsers.add_parser("start")
    start.add_argument("--device-id", default="device-local-01")
    start.add_argument("--text", required=True)
    start.add_argument("--exact-input-text")
    start.add_argument("--exact-action-kind", choices=("back", "home", "tap_semantic"))
    start.add_argument("--exact-target-label", default="")
    start.add_argument("--auto-advance", action="store_true")
    for name in ("get", "confirm-once", "next", "cancel", "pause"):
        command = subparsers.add_parser(name)
        command.add_argument("--session-id", required=True)
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
            elif args.command == "start":
                raw = client.start_session(
                    text=args.text,
                    exact_input_text=args.exact_input_text,
                    exact_action_kind=args.exact_action_kind,
                    exact_target_label=args.exact_target_label,
                    device_id=args.device_id,
                    auto_advance=args.auto_advance,
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
