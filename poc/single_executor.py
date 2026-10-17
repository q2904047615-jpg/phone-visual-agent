"""Composition root: one Runtime owner behind any number of API ingresses."""
from __future__ import annotations

import os
from pathlib import Path

from agent.infrastructure.command_inbox import CommandInbox, ExecutorInboxMiddleware
from agent.infrastructure.device_exclusivity import InterProcessLease


def create_app():
    directory = Path(os.environ["PHONE_AGENT_STATE_DIR"]).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    lease = InterProcessLease(directory / "executor.lock", owner_id="runtime-executor",
                              metadata={"role": "single-executor"})
    if not lease.acquire():
        raise RuntimeError("an executor already owns this state directory")
    try:
        inbox = CommandInbox(directory / "dispatch.sqlite3")
        inbox.recover()
        inbox.prune()
        # Redirect all Runtime output before importing the existing composition
        # root. API workers never import this module or initialize a controller.
        from agent.infrastructure import robot_controller
        robot_controller.WEB_OUTPUT_DIR = directory
        os.environ["PHONE_AGENT_EXECUTOR_MODE"] = "1"
        import web_app
        from agent.infrastructure.durable_async_tasks import DurableAsyncTaskRegistry
        old_tasks = web_app._GENERIC_START_TASKS
        web_app._GENERIC_START_TASKS = DurableAsyncTaskRegistry(
            directory / "dispatch.sqlite3", ttl_seconds=old_tasks.ttl_seconds,
            max_tasks=old_tasks.max_tasks)
        old_tasks.shutdown()
        inner = ExecutorInboxMiddleware(web_app.app, inbox)
    except BaseException:
        lease.release()
        raise

    async def application(scope, receive, send):
        if scope["type"] == "lifespan":
            try:
                await inner(scope, receive, send)
            finally:
                lease.release()
        else:
            await inner(scope, receive, send)
    return application
