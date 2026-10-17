"""Stateless ASGI ingress. Never imports web_app or constructs a Runtime."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from agent.infrastructure.command_inbox import CommandInbox, is_command, request_fingerprint

HOP_HEADERS = {b"connection", b"keep-alive", b"proxy-authenticate", b"proxy-authorization",
               b"te", b"trailer", b"transfer-encoding", b"upgrade"}


def forwarded_headers(headers):
    connection = set()
    for key, value in headers:
        if key.lower() == b"connection":
            connection.update(item.strip().lower() for item in value.split(b","))
    return [(k, v) for k, v in headers if k.lower() not in HOP_HEADERS | connection]


def create_app():
    target = os.environ["PHONE_AGENT_EXECUTOR_URL"].rstrip("/")
    parsed = urlsplit(target)
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or not parsed.port
            or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment):
        raise ValueError("executor must be an explicit loopback HTTP host and port")
    inbox = CommandInbox(Path(os.environ["PHONE_AGENT_STATE_DIR"]) / "dispatch.sqlite3")

    @asynccontextmanager
    async def lifespan(app):
        # Streaming camera responses may be long-lived. No implicit retries.
        async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=5),
                                     trust_env=False, follow_redirects=False) as client:
            app.state.client = client
            yield

    app = FastAPI(lifespan=lifespan, openapi_url=None, docs_url=None, redoc_url=None)

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
    async def forward(request: Request):
        body = await request.body()
        headers = [(k, v) for k, v in forwarded_headers(request.scope["headers"])
                   if k.lower() not in {b"x-agent-dispatch-ticket", b"x-forwarded-for",
                                        b"x-forwarded-proto", b"x-forwarded-host"}]
        # Scheme comes from ASGI's explicitly trusted reverse proxy, never
        # from arbitrary incoming forwarded headers. Original Host is retained.
        headers.append((b"x-forwarded-proto", request.scope["scheme"].encode()))
        ticket = None
        if is_command(request.scope):
            fingerprint = request_fingerprint(request.method, request.scope["path"],
                                              request.scope["query_string"], headers, body)
            ticket = inbox.enqueue(fingerprint)
            headers.append((b"x-agent-dispatch-ticket", ticket.encode()))
        raw_path = request.scope.get("raw_path", request.url.path.encode())
        query = request.scope["query_string"]
        url = target + raw_path.decode("ascii") + ("?" + query.decode("ascii") if query else "")
        try:
            upstream = await app.state.client.send(
                app.state.client.build_request(request.method, url, headers=headers, content=body),
                stream=True)
        except httpx.ConnectError:
            if ticket:
                inbox.finish(ticket, "not_dispatched")
            return JSONResponse({"detail": {"code": "executor_unavailable", "dispatched": False}},
                                status_code=503)
        except httpx.HTTPError:
            # A lost response does not prove the command did not run.
            return JSONResponse({"detail": {"code": "executor_response_unknown",
                                            "dispatch_ticket": ticket}}, status_code=502)

        async def stream():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()
        response = StreamingResponse(stream(), status_code=upstream.status_code,
                                     background=BackgroundTask(upstream.aclose))
        response.raw_headers = forwarded_headers(upstream.headers.raw)
        return response

    return app
