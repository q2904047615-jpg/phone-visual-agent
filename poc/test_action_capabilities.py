import unittest

from action_capabilities import (
    ActionCapabilityError,
    KNOWN_ACTION_CAPABILITIES,
    build_device_capability_snapshot,
)


class ActionCapabilityTests(unittest.TestCase):
    def test_snapshot_fills_every_known_action_and_stable_digest(self):
        first = build_device_capability_snapshot(
            device_id="device-1",
            supported_actions={"tap_semantic", "swipe", "input_verified_text"},
            raw_profile={
                "actions": {
                    "input_verified_text": {
                        "canonical_max_chars": 4000,
                        "newline": "requires_fresh_visible_enter_key",
                    }
                }
            },
        )
        second = build_device_capability_snapshot(
            device_id="device-1",
            supported_actions={"input_verified_text", "swipe", "tap_semantic"},
            raw_profile={
                "actions": {
                    "input_verified_text": {
                        "newline": "requires_fresh_visible_enter_key",
                        "canonical_max_chars": 4000,
                    }
                }
            },
        )
        self.assertEqual(KNOWN_ACTION_CAPABILITIES, set(first.actions))
        self.assertEqual(first.profile_digest, second.profile_digest)
        self.assertEqual(4000, first.actions["input_verified_text"]["canonical_max_chars"])

    def test_unsupported_double_tap_returns_typed_gap(self):
        snapshot = build_device_capability_snapshot(
            device_id="device-1",
            supported_actions={"tap_semantic"},
            raw_profile={
                "actions": {
                    "double_tap": {
                        "transport": "seller_click_count_two_atomic_request",
                        "gap_reason": "requires_double_tap_live_acceptance",
                    }
                }
            },
        )
        gap = snapshot.gap("double_tap", required_parameters=("interval_ms",))
        self.assertIsNotNone(gap)
        self.assertEqual("requires_double_tap_live_acceptance", gap.reason_code)
        self.assertEqual(("interval_ms",), gap.required_parameters)
        self.assertRegex(gap.profile_digest, r"^[0-9a-f]{64}$")

    def test_supported_action_has_no_gap_and_unknown_action_is_rejected(self):
        snapshot = build_device_capability_snapshot(
            device_id="device-1",
            supported_actions={"tap_semantic"},
        )
        self.assertIsNone(snapshot.gap("tap_semantic"))
        with self.assertRaises(ActionCapabilityError):
            snapshot.gap("app_specific_magic")


if __name__ == "__main__":
    unittest.main()
