from __future__ import annotations

import json
import unittest

from agent.application.entry_router import (
    AGENT_ENTRY_ROUTE_PROTOCOL_VERSION,
    AgentEntryRouteError,
    AgentEntryRouter,
)


class _Gateway:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []
        self.scopes = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.payload

    def _chat(self, *args, **kwargs):  # pragma: no cover - proves private transport is not used
        raise AssertionError("entry router must use the public Model Gateway chat contract")

    def call_scope(self, *, stage):
        from contextlib import contextmanager

        @contextmanager
        def scope():
            self.scopes.append(stage)
            yield

        return scope()


class AgentEntryRouterTests(unittest.TestCase):
    def test_routes_through_public_gateway_and_keeps_runtime_authority_out(self):
        gateway = _Gateway(json.dumps({
            "route": "phone_task",
            "reply": "准备把目标交给手机 Runtime。",
            "reason": "用户明确要求操作手机。",
        }))
        result = AgentEntryRouter(gateway).route(
            text="打开手机设置",
            conversation=[{"role": "user", "content": "开始"}],
        )
        self.assertEqual("phone_task", result.route)
        self.assertEqual(["agent_entry_route"], gateway.scopes)
        self.assertEqual(1, len(gateway.calls))
        messages, kwargs = gateway.calls[0]
        self.assertEqual("system", messages[0]["role"])
        self.assertEqual("user", messages[-1]["role"])
        self.assertEqual("object", kwargs["response_format"]["json_schema"]["schema"]["type"])
        self.assertEqual(AGENT_ENTRY_ROUTE_PROTOCOL_VERSION, result.to_dict()["protocol_version"])

    def test_rejects_malformed_or_unknown_gateway_route(self):
        with self.assertRaises(AgentEntryRouteError):
            AgentEntryRouter(_Gateway("not-json")).route(text="你好")
        with self.assertRaises(AgentEntryRouteError):
            AgentEntryRouter(_Gateway(json.dumps({
                "route": "finish",
                "reply": "",
                "reason": "",
            }))).route(text="你好")

    def test_requires_public_chat_contract(self):
        class DecisionOnlyGateway:
            def decide(self, **kwargs):
                return kwargs

        with self.assertRaises(AgentEntryRouteError):
            AgentEntryRouter(DecisionOnlyGateway()).route(text="你好")


if __name__ == "__main__":
    unittest.main()
