import unittest
from types import SimpleNamespace

from qwen_visual_decision import _deterministic_exact_selection_payload


class DeterministicExactSelectionTests(unittest.TestCase):
    def test_exact_subgoal_selects_only_candidate_without_model(self) -> None:
        context = SimpleNamespace(
            current_subgoal={"subgoal_id": "exact_tap_semantic"}
        )
        payload = _deterministic_exact_selection_payload(
            context,
            ({"choice_id": "choice_1", "action": "tap_semantic"},),
        )
        self.assertEqual("choice_1", payload["choice_id"])
        self.assertEqual("action", payload["status"])

    def test_exact_subgoal_selects_unique_matching_action_among_other_kinds(self) -> None:
        context = SimpleNamespace(
            current_subgoal={"subgoal_id": "exact_tap_semantic"}
        )
        payload = _deterministic_exact_selection_payload(
            context,
            (
                {"choice_id": "choice_1", "action": "back"},
                {"choice_id": "choice_2", "action": "tap_semantic"},
            ),
        )
        self.assertEqual("choice_2", payload["choice_id"])

    def test_exact_tap_uses_unique_local_semantic_target_among_other_taps(self) -> None:
        context = SimpleNamespace(
            current_subgoal={"subgoal_id": "exact_tap_semantic"}
        )
        observation = SimpleNamespace(
            target_local_candidate=lambda: SimpleNamespace(
                element_id="local_audited_reload_control_1"
            )
        )

        payload = _deterministic_exact_selection_payload(
            context,
            (
                {
                    "choice_id": "choice_verify",
                    "action": "tap_semantic",
                    "element_id": "verify-button",
                },
                {
                    "choice_id": "choice_reload",
                    "action": "tap_semantic",
                    "element_id": "local_audited_reload_control_1",
                },
            ),
            observation=observation,
        )

        self.assertEqual("choice_reload", payload["choice_id"])

    def test_exact_tap_does_not_invent_choice_for_unlisted_local_target(self) -> None:
        context = SimpleNamespace(
            current_subgoal={"subgoal_id": "exact_tap_semantic"}
        )
        observation = SimpleNamespace(
            target_local_candidate=lambda: SimpleNamespace(
                element_id="local_target_without_choice"
            )
        )

        self.assertIsNone(
            _deterministic_exact_selection_payload(
                context,
                (
                    {
                        "choice_id": "choice_other",
                        "action": "tap_semantic",
                        "element_id": "other-button",
                    },
                ),
                observation=observation,
            )
        )

    def test_non_exact_or_same_action_ambiguous_catalog_still_requires_model(self) -> None:
        regular = SimpleNamespace(current_subgoal={"subgoal_id": "navigate"})
        exact = SimpleNamespace(
            current_subgoal={"subgoal_id": "exact_tap_semantic"}
        )
        self.assertIsNone(
            _deterministic_exact_selection_payload(
                regular,
                ({"choice_id": "choice_1"},),
            )
        )
        self.assertIsNone(
            _deterministic_exact_selection_payload(
                exact,
                (
                    {"choice_id": "choice_1", "action": "tap_semantic"},
                    {"choice_id": "choice_2", "action": "tap_semantic"},
                ),
            )
        )

    def test_exact_input_selects_only_current_local_target_candidate(self) -> None:
        context = SimpleNamespace(
            current_subgoal={"subgoal_id": "input_exact_text"}
        )
        observation = SimpleNamespace(
            target_local_candidate=lambda: SimpleNamespace(element_id="field")
        )

        payload = _deterministic_exact_selection_payload(
            context,
            (
                {
                    "choice_id": "choice_1",
                    "action": "input_verified_text",
                    "element_id": "field",
                },
                {
                    "choice_id": "choice_2",
                    "action": "tap_semantic",
                    "element_id": "mode-switch",
                },
            ),
            observation=observation,
        )

        self.assertEqual("choice_1", payload["choice_id"])

    def test_exact_input_stays_model_free_across_newline_microstep(self) -> None:
        context = SimpleNamespace(
            current_subgoal={"subgoal_id": "input_exact_text"}
        )
        observation = SimpleNamespace(
            target_local_candidate=lambda: SimpleNamespace(element_id="enter-key")
        )

        payload = _deterministic_exact_selection_payload(
            context,
            (
                {
                    "choice_id": "choice_1",
                    "action": "press_enter",
                    "element_id": "enter-key",
                },
            ),
            observation=observation,
        )

        self.assertEqual("choice_1", payload["choice_id"])

    def test_exact_input_requires_one_canonical_choice_for_local_target(self) -> None:
        context = SimpleNamespace(
            current_subgoal={"subgoal_id": "input_exact_text"}
        )
        missing_target = SimpleNamespace(target_local_candidate=lambda: None)
        ambiguous_target = SimpleNamespace(
            target_local_candidate=lambda: SimpleNamespace(element_id="field")
        )

        self.assertIsNone(
            _deterministic_exact_selection_payload(
                context,
                ({"choice_id": "choice_1", "action": "input_verified_text"},),
                observation=missing_target,
            )
        )
        self.assertIsNone(
            _deterministic_exact_selection_payload(
                context,
                (
                    {
                        "choice_id": "choice_1",
                        "action": "input_verified_text",
                        "element_id": "field",
                    },
                    {
                        "choice_id": "choice_2",
                        "action": "clear_verified_text",
                        "element_id": "field",
                    },
                ),
                observation=ambiguous_target,
            )
        )

    def test_exact_input_selects_clear_when_current_value_is_not_a_prefix(self) -> None:
        context = SimpleNamespace(
            current_subgoal={"subgoal_id": "input_exact_text"},
            requested_input_text="x\ny",
        )
        observation = SimpleNamespace(
            target_local_candidate=lambda: SimpleNamespace(
                element_id="field",
                states={"value": "first\n"},
            )
        )

        payload = _deterministic_exact_selection_payload(
            context,
            (
                {
                    "choice_id": "choice_1",
                    "action": "clear_verified_text",
                    "element_id": "field",
                },
                {
                    "choice_id": "choice_2",
                    "action": "tap_semantic",
                    "element_id": "mode-switch",
                },
            ),
            observation=observation,
        )

        self.assertEqual("choice_1", payload["choice_id"])


if __name__ == "__main__":
    unittest.main()
