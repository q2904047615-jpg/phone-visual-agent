"""Durable dispatch tickets; never a second Task/Run lifecycle authority.

Only request fingerprints are retained, not cookies, tokens or user content.
After a crash, tickets are classified, never replayed: a new request must
pass normal API authentication and Runtime recovery again.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


def request_fingerprint(method: str, path: str, query: bytes, headers, body: bytes) -> str:
    # Bind the transient authorization too, without retaining its value.
    bound = sorted((k.lower().decode("latin1"), v.decode("latin1")) for k, v in headers
                   if k.lower() in {b"cookie", b"authorization", b"x-control-token",
                                    b"x-control-plane-token", b"idempotency-key", b"content-type"})
    digest = hashlib.sha256(json.dumps([method, path, query.hex(), bound],
                                      separators=(",", ":")).encode())
    digest.update(body)
    return digest.hexdigest()


class CommandInbox:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS dispatch_tickets (
                ticket_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
                status TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL,
                http_status INTEGER)""")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def enqueue(self, fingerprint: str) -> str:
        ticket = uuid4().hex
        now = time.time()
        with self.connect() as db:
            db.execute("INSERT INTO dispatch_tickets VALUES(?,?,?,?,?,NULL)",
                       (ticket, fingerprint, "queued", now, now))
        return ticket

    def claim(self, ticket: str, fingerprint: str) -> bool:
        with self.connect() as db:
            return db.execute("UPDATE dispatch_tickets SET status='dispatching',updated_at=? "
                              "WHERE ticket_id=? AND fingerprint=? AND status='queued'",
                              (time.time(), ticket, fingerprint)).rowcount == 1

    def finish(self, ticket: str, status: str, http_status: int | None = None):
        if status not in {"responded", "unknown", "not_dispatched"}:
            raise ValueError("invalid dispatch outcome")
        expected = "queued" if status == "not_dispatched" else "dispatching"
        with self.connect() as db:
            db.execute("UPDATE dispatch_tickets SET status=?,http_status=?,updated_at=? "
                       "WHERE ticket_id=? AND status=?",
                       (status, http_status, time.time(), ticket, expected))

    def recover(self):
        # Called only after acquiring the single executor OS lock, before bind.
        with self.connect() as db:
            db.execute("UPDATE dispatch_tickets SET status=CASE status "
                       "WHEN 'queued' THEN 'not_dispatched' ELSE 'unknown' END,updated_at=? "
                       "WHERE status IN ('queued','dispatching')", (time.time(),))

    def prune(self, retention_seconds: float = 14 * 86400):
        with self.connect() as db:
            db.execute("DELETE FROM dispatch_tickets WHERE status IN "
                       "('responded','not_dispatched') AND updated_at < ?",
                       (time.time() - retention_seconds,))


def is_command(scope) -> bool:
    return (scope["method"] not in {"GET", "HEAD", "OPTIONS"}
            and scope["path"].startswith("/api/")
            and not scope["path"].startswith("/api/control-plane/auth/"))


class ExecutorInboxMiddleware:
    def __init__(self, app, inbox: CommandInbox):
        self.app, self.inbox = app, inbox

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not is_command(scope):
            return await self.app(scope, receive, send)
        from starlette.responses import JSONResponse
        chunks = []
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        ticket = dict(scope["headers"]).get(b"x-agent-dispatch-ticket", b"").decode("ascii", "replace")
        fingerprint = request_fingerprint(scope["method"], scope["path"],
                                          scope["query_string"], scope["headers"], body)
        if not self.inbox.claim(ticket, fingerprint):
            return await JSONResponse({"detail": {"code": "dispatch_ticket_invalid"}},
                                      status_code=409)(scope, receive, send)
        received = False
        status = None
        async def replay_receive():
            nonlocal received
            if not received:
                received = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()
        async def tracked_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message = dict(message, headers=list(message.get("headers", [])) +
                               [(b"x-agent-dispatch-ticket", ticket.encode())])
            # Commit receipt before notifying client of the final response.
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                self.inbox.finish(ticket, "responded", status)
            await send(message)
        try:
            await self.app(scope, replay_receive, tracked_send)
        finally:
            # No-op if a complete response was already recorded. HTTP 200 is
            # only a response receipt, never a business-success assertion.
            self.inbox.finish(ticket, "unknown")
