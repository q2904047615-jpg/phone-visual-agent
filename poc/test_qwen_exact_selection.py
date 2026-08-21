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


if __name__ == "__main__":
    unittest.main()
