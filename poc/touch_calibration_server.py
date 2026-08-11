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
        if path == "/api/samples":
            self._send_json(STORE.snapshot())
            return
        if path == "/api/health":
            self._send_json({"ok": True, "service": "touch-calibration"})
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
