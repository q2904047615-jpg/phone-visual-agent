from __future__ import annotations

import argparse
import json
import math
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
ROOT_ACTION_MODES = frozenset(
    {"index", "swipe", "tap", "back", "input", "long_press", "drag", "sequence"}
)


def root_action_location(mode: str | None) -> str | None:
    normalized = str(mode or "").strip()
    if not normalized:
        return None
    if normalized not in ROOT_ACTION_MODES:
        raise ValueError(f"不支持的根页动作验收模式: {normalized}")
    return f"/actions?mode={normalized}"


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


class PageStateStore:
    ALLOWED_PHASES = frozenset(
        {"fullscreen_setup", "calibration", "complete", "blocked"}
    )
    ALLOWED_MODES = frozenset(
        {"setup", "fullscreen", "viewport_coverage", "blocked"}
    )

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.state: dict[str, object] = {
            "phase": "unknown",
            "fullscreen": False,
        }

    def update(self, payload: dict[str, object]) -> dict[str, object]:
        phase = str(payload.get("phase") or "")
        if phase not in self.ALLOWED_PHASES:
            raise ValueError(f"不支持的校准页阶段: {phase or 'missing'}")
        width = int(payload.get("viewport_width") or 0)
        height = int(payload.get("viewport_height") or 0)
        sequence = int(payload.get("sequence") or 0)
        if width < 1 or height < 1 or not 0 <= sequence <= 9:
            raise ValueError("校准页状态尺寸或序号无效")
        fullscreen = payload.get("fullscreen") is True
        mode = str(payload.get("calibration_mode") or "")
        attempted = payload.get("fullscreen_attempted") is True
        coverage = payload.get("viewport_coverage")
        if mode not in self.ALLOWED_MODES:
            raise ValueError(f"不支持的校准模式: {mode or 'missing'}")
        if not isinstance(coverage, dict):
            raise ValueError("缺少视口覆盖证据")
        width_ratio = coverage.get("width_ratio", 0)
        height_ratio = coverage.get("height_ratio", 0)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in (width_ratio, height_ratio)
        ):
            raise ValueError("视口覆盖比例必须是数值")
        coverage_evidence = {
            "eligible": coverage.get("eligible") is True,
            "width_ratio": float(width_ratio),
            "height_ratio": float(height_ratio),
        }
        if not all(
            math.isfinite(coverage_evidence[name])
            for name in ("width_ratio", "height_ratio")
        ):
            raise ValueError("视口覆盖比例无效")
        if phase == "fullscreen_setup" and (mode != "setup" or fullscreen):
            raise ValueError("全屏准备阶段状态不一致")
        if phase in {"calibration", "complete"}:
            fullscreen_ready = mode == "fullscreen" and fullscreen
            viewport_ready = (
                mode == "viewport_coverage"
                and not fullscreen
                and attempted
                and coverage_evidence["eligible"]
                and 0.92 <= coverage_evidence["width_ratio"] <= 1.08
                and 0.92 <= coverage_evidence["height_ratio"] <= 1.08
            )
            if not (fullscreen_ready or viewport_ready):
                raise ValueError("校准阶段缺少可信全屏或高覆盖视口证据")
        if phase == "blocked" and (mode != "blocked" or fullscreen or not attempted):
            raise ValueError("阻断阶段状态不一致")
        with self.lock:
            self.state = {
                "phase": phase,
                "fullscreen": fullscreen,
                "calibration_mode": mode,
                "fullscreen_attempted": attempted,
                "fullscreen_method": str(payload.get("fullscreen_method") or "")[:80],
                "viewport_coverage": coverage_evidence,
                "error": str(payload.get("error") or "")[:240],
                "viewport_width": width,
                "viewport_height": height,
                "sequence": sequence,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            return dict(self.state)

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return dict(self.state)


PAGE_STATE = PageStateStore()


class ActionEventStore:
    ALLOWED_KINDS = frozenset(
        {
            "swipe",
            "tap_semantic",
            "back",
            "input_verified_text",
            "long_press",
            "drag",
        }
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
    root_action_mode: str | None = None

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
            redirect = root_action_location(self.root_action_mode)
            if redirect is not None:
                self.send_response(HTTPStatus.SEE_OTHER)
                self.send_header("Location", redirect)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
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
        if path == "/api/page-state":
            self._send_json(PAGE_STATE.snapshot())
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
            if path == "/api/page-state":
                self._send_json({"ok": True, **PAGE_STATE.update(payload)})
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
    parser.add_argument(
        "--root-action-mode",
        choices=sorted(ROOT_ACTION_MODES),
        default=None,
        help="让根页安全重定向到指定通用动作验收模式；省略时仍为校准页",
    )
    args = parser.parse_args()
    Handler.root_action_mode = args.root_action_mode
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
