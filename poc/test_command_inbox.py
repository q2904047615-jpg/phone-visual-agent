from pathlib import Path
import asyncio
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

from agent.infrastructure.command_inbox import CommandInbox, ExecutorInboxMiddleware, request_fingerprint


class CommandInboxTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "inbox.sqlite3"
        self.inbox = CommandInbox(self.path)

    def status(self, ticket):
        with self.inbox.connect() as db:
            return db.execute("SELECT status FROM dispatch_tickets WHERE ticket_id=?", (ticket,)).fetchone()[0]

    def test_competing_claims_only_dispatch_once(self):
        ticket = self.inbox.enqueue("fp")
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(lambda _: self.inbox.claim(ticket, "fp"), range(2)))
        self.assertEqual([False, True], sorted(results))
        self.inbox.finish(ticket, "responded", 409)
        self.inbox.finish(ticket, "unknown")
        self.assertEqual("responded", self.status(ticket))

    def test_restart_distinguishes_queued_from_unknown_and_never_replays(self):
        queued = self.inbox.enqueue("before")
        running = self.inbox.enqueue("after")
        self.assertTrue(self.inbox.claim(running, "after"))
        restarted = CommandInbox(self.path)
        restarted.recover()
        self.assertEqual("not_dispatched", self.status(queued))
        self.assertEqual("unknown", self.status(running))
        self.assertFalse(restarted.claim(running, "after"))
        self.assertFalse(restarted.claim(queued, "before"))

    def test_fingerprint_binds_identity_content_and_idempotency_without_storing_secrets(self):
        def fp(token=b"private-cookie", body=b"private-goal"):
            return request_fingerprint("POST", "/api/agent/chat", b"", [(b"cookie", token)], body)
        self.assertNotEqual(fp(), fp(b"different-user"))
        self.assertNotEqual(fp(), fp(body=b"changed-command"))
        self.inbox.enqueue(fp())
        with self.inbox.connect() as db:
            dump = "\n".join(db.iterdump())
        self.assertNotIn("private-cookie", dump)
        self.assertNotIn("private-goal", dump)

    def test_fingerprint_mismatch_does_not_consume_ticket(self):
        ticket = self.inbox.enqueue("original")
        self.assertFalse(self.inbox.claim(ticket, "changed"))
        self.assertTrue(self.inbox.claim(ticket, "original"))

    def test_dispatch_exception_becomes_unknown_without_automatic_retry(self):
        scope = {"type":"http", "method":"POST", "path":"/api/agent/chat", "query_string":b"", "headers":[]}
        ticket = self.inbox.enqueue(request_fingerprint("POST", scope["path"], b"", [], b"{}"))
        scope["headers"] = [(b"x-agent-dispatch-ticket", ticket.encode())]
        calls = []
        async def app(scope, receive, send):
            calls.append(1)
            raise RuntimeError("simulated executor interruption")
        async def receive():
            return {"type":"http.request", "body":b"{}"}
        async def send(message):
            pass
        wrapper = ExecutorInboxMiddleware(app, self.inbox)
        with self.assertRaises(RuntimeError):
            asyncio.run(wrapper(scope, receive, send))
        asyncio.run(wrapper(scope, receive, send))
        self.assertEqual([1], calls)
        self.assertEqual("unknown", self.status(ticket))
