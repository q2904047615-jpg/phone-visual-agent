"""Bounded public HTTPS read probe through the project's official API client."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_agent_api_client import LocalAgentApiClient, LocalAgentApiError


class PublicReadTransport(httpx.BaseTransport):
    def __init__(self, origin):
        parsed = urlsplit(origin)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
            raise ValueError("A bare HTTPS origin is required")
        self.origin = httpx.URL(origin)
        self.inner = httpx.HTTPTransport(retries=0)

    def handle_request(self, request):
        # bootstrap verifies the running OpenAPI; the protected read is a
        # negative auth check, never a forged principal or a device request.
        if request.method != "GET" or request.url.path not in {
            "/api/session", "/openapi.json", "/api/control-plane/organizations/local-org",
        }:
            raise ValueError("Only the acceptance read contract is allowed")
        request.url = request.url.copy_with(scheme="https", host=self.origin.host, port=self.origin.port)
        request.headers["host"] = self.origin.netloc.decode()
        return self.inner.handle_request(request)

    def close(self):
        self.inner.close()


def probe(origin, *, clients=4, rounds=150, interval=2.0):
    if not 1 <= clients <= 4 or not 1 <= rounds <= 300 or not 1 <= interval <= 10:
        raise ValueError("Probe bounds exceeded")
    started = datetime.now(timezone.utc).isoformat()
    def worker(_):
        rows = []
        for index in range(rounds):
            begin = time.monotonic()
            row = {"ok": False}
            try:
                with LocalAgentApiClient(transport=PublicReadTransport(origin), timeout_seconds=10) as api:
                    row["openapi_verified"] = api.bootstrap()["openapi_verified"]
                    if index % 10 == 0:
                        try:
                            api.control_plane_organization("local-org")
                            row["error"] = "anonymous_read_unexpectedly_allowed"
                        except LocalAgentApiError as exc:
                            row["anonymous_denied"] = exc.details.status_code == 401
                            if not row["anonymous_denied"]:
                                row["error"] = "unexpected_auth_status"
                    row["ok"] = row["openapi_verified"] is True and "error" not in row
            except Exception as exc:
                # Never print response bodies, cookies, tokens or exception text.
                row["error"] = type(exc).__name__
            row["duration_ms"] = round((time.monotonic() - begin) * 1000, 2)
            rows.append(row)
            if not row["ok"]:
                break  # Stop the affected client instead of hammering an error.
            if index + 1 < rounds:
                time.sleep(interval)
        return rows
    with ThreadPoolExecutor(clients) as pool:
        results = list(pool.map(worker, range(clients)))
    rows = [row for group in results for row in group]
    durations = sorted(row["duration_ms"] for row in rows)
    return {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "origin": origin, "clients": clients, "planned_rounds": clients * rounds,
            "completed_rounds": len(rows), "passed_rounds": sum(row["ok"] for row in rows),
            "anonymous_denials": sum(row.get("anonymous_denied", False) for row in rows),
            "cycle_p50_ms": durations[len(durations)//2],
            "cycle_p95_ms": durations[min(len(durations)-1, int(len(durations)*.95))],
            "cycle_max_ms": max(durations), "errors": [row["error"] for row in rows if not row["ok"]],
            "scope": "public_tls_bootstrap_and_anonymous_denial_only",
            "status": "passed" if len(rows) == clients * rounds and all(r["ok"] for r in rows) else "failed"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", required=True)
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = probe(args.origin, clients=args.clients, rounds=args.rounds, interval=args.interval)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report))
    raise SystemExit(0 if report["status"] == "passed" else 1)
