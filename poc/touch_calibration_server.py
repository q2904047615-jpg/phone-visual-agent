from __future__ import annotations

import argparse
import json
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
PAGE_PATH = ROOT / "static" / "touch_calibration.html"
ACTION_PAGE_PATH = ROOT / "static" / "action_acceptance.html"
OUTPUT_ROOT = ROOT / "output" / "xy_calibration"


class SampleStore:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.samples: list[dict[str, object]] = []
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return OUTPUT_ROOT / f"browser_touches_{self.session_id}.jsonl"

    def reset(self) -> None:
        with self.lock:
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.samples = []

    def append(self, payload: dict[str, object]) -> dict[str, object]:
        required = {"sequence", "target_x", "target_y", "actual_x", "actual_y", "viewport_width", "viewport_height"}
        if not required.issubset(payload):
            missing = ", ".join(sorted(required - payload.keys()))
            raise ValueError(f"缺少触点字段: {missing}")
        with self.lock:
            record = dict(payload)
            record["session_id"] = self.session_id
            record["received_at"] = datetime.now(timezone.utc).isoformat()
            self.samples.append(record)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            return record

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return {"session_id": self.session_id, "samples": list(self.samples)}


STORE = SampleStore()


class ActionEventStore:
    ALLOWED_KINDS = frozenset(
        {"swipe", "back", "input_verified_text", "long_press", "drag"}
    )
    ALLOWED_STATUSES = frozenset({"passed", "failed"})

    def __init__(self, output_root: Path = OUTPUT_ROOT) -> None:
        self.lock = threading.Lock()
        self.output_root = Path(output_root)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.events: list[dict[str, object]] = []
        self.event_ids: set[str] = set()
        self.output_root.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self.output_root / f"action_events_{self.session_id}.jsonl"

    def reset(self) -> None:
        with self.lock:
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.events = []
            self.event_ids = set()

    def append(self, payload: dict[str, object]) -> dict[str, object]:
        if not isinstance(payload, dict):
            raise ValueError("动作事件必须是JSON对象。")
        kind = str(payload.get("kind") or "").strip()
        status = str(payload.get("status") or "").strip()
        client_event_id = str(payload.get("client_event_id") or "").strip()
        details = payload.get("details", {})
        if kind not in self.ALLOWED_KINDS:
            raise ValueError(f"不支持的动作事件: {kind or 'missing'}")
        if status not in self.ALLOWED_STATUSES:
            raise ValueError(f"不支持的动作状态: {status or 'missing'}")
        if not client_event_id or len(client_event_id) > 128:
            raise ValueError("client_event_id缺失或过长。")
        if not isinstance(details, dict):
            raise ValueError("details必须是JSON对象。")
        with self.lock:
            if client_event_id in self.event_ids:
                raise ValueError("重复的client_event_id。")
            record = {
                "sequence": len(self.events),
                "kind": kind,
                "status": status,
                "client_event_id": client_event_id,
                "details": dict(details),
                "session_id": self.session_id,
                "received_at": datetime.now(timezone.utc).isoformat(),
            }
            self.events.append(record)
            self.event_ids.add(client_event_id)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            return record

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return {"session_id": self.session_id, "events": list(self.events)}


ACTION_STORE = ActionEventStore()


class Handler(BaseHTTPRequestHandler):
    server_version = "TouchCalibration/1.0"

    def _send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlparse(self.path).path
        if path == "/":
            body = PAGE_PATH.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/actions":
            body = ACTION_PAGE_PATH.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/samples":
            self._send_json(STORE.snapshot())
            return
        if path == "/api/action-events":
            self._send_json(ACTION_STORE.snapshot())
            return
        if path == "/api/health":
            self._send_json(
                {
                    "ok": True,
                    "service": "touch-calibration",
                    "action_acceptance": True,
                }
            )
            return
        self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 64_000:
                raise ValueError("请求过大")
            payload = json.loads(self.rfile.read(length) or b"{}")
            if path == "/api/sample":
                self._send_json({"ok": True, "sample": STORE.append(payload)})
                return
            if path == "/api/reset":
                STORE.reset()
                self._send_json({"ok": True, **STORE.snapshot()})
                return
            if path == "/api/action-event":
                self._send_json(
                    {"ok": True, "event": ACTION_STORE.append(payload)}
                )
                return
            if path == "/api/action-reset":
                ACTION_STORE.reset()
                self._send_json({"ok": True, **ACTION_STORE.snapshot()})
                return
            self._send_json({"error": "not found"}, 404)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, 400)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="Harmless multi-point touch calibration page")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Touch calibration: http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
