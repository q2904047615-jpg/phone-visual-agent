from __future__ import annotations

import unittest

from agent.domain.action_catalog import CANONICAL_ACTION_KINDS
from agent.domain.model_gateway import ModelGatewayRegistration, ModelGatewayRegistry
from agent.domain.tool_registry import ToolRegistry, ToolRegistration, build_canonical_tool_registry


class _Gateway:
    def decide(self, **kwargs):
        return kwargs

    def status(self):
        return {"configured": True}


class GatewayAndToolRegistryTests(unittest.TestCase):
    def test_gateway_registry_selects_default_and_rejects_duplicate(self):
        gateway = _Gateway()
        registration = ModelGatewayRegistration(
            gateway_id="test", model="test-model", capabilities=frozenset({"vision"}),
            version="v1", gateway=gateway,
        )
        registry = ModelGatewayRegistry()
        registry.register(registration, default=True)
        self.assertIs(registry.default(), gateway)
        with self.assertRaises(ValueError):
            registry.register(registration)
        with self.assertRaises(KeyError):
            registry.require_registration("missing")

    def test_canonical_tool_registry_covers_only_one_action_catalog(self):
        registry = build_canonical_tool_registry()
        self.assertEqual(registry.names(), CANONICAL_ACTION_KINDS)
        self.assertEqual(registry.require("tap_semantic").idempotency, "single_step_no_replay")
        with self.assertRaises(KeyError):
            registry.require("unknown_tool")

    def test_disabled_tool_is_rejected_before_execution(self):
        registry = ToolRegistry([ToolRegistration(
            name="tap_semantic", version="v1", effect_category="device_physical",
            permissions=frozenset({"device.execute"}), idempotency="single_step_no_replay",
            rollback="reobserve_or_stop", simulation_supported=True, real_supported=True, enabled=False,
        )])
        with self.assertRaises(ValueError):
            registry.require("tap_semantic")


if __name__ == "__main__":
    unittest.main()
