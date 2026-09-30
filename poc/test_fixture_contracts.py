"""Test helpers must not rewrite requests at the execution boundary."""
import unittest
from agent.domain.semantic_action import SemanticAction
from agent.domain.universal_action_controller import UniversalActionController
from agent.infrastructure.generic_action_adapter import GenericSingleActionAdapter
from test_support import generic_action_adapter, ui_scene


class ExplicitFixtureContractTests(unittest.TestCase):
    def test_adapter_fixture_inherits_real_execute_without_request_rewrite(self):
        self.assertIs(GenericSingleActionAdapter.execute,
                      generic_action_adapter.GenericSingleActionAdapter.execute)

    def test_controller_fixture_is_real_controller_without_parameter_rewrite(self):
        self.assertIs(UniversalActionController, ui_scene.UniversalActionController)

    def test_action_fixture_is_real_action_without_default_click_point(self):
        self.assertIs(SemanticAction, generic_action_adapter.SemanticAction)


if __name__ == '__main__':
    unittest.main()
