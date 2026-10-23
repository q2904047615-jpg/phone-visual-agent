"""Retired standalone probes must not return as runnable project tools."""
import ast
from pathlib import Path
import unittest

POC = Path(__file__).resolve().parents[2]
RETIRED = frozenset({
    'qwen_action_union_contract', 'qwen_strict_direct_point_contract',
    'qwen_flat_contract_probe', 'audit_point_scene_contract', 'probe_runtime_restrictions',
})


class RetiredExperimentToolsTests(unittest.TestCase):
    def test_retired_entry_files_are_absent(self):
        for name in RETIRED:
            with self.subTest(tool=name):
                self.assertFalse((POC / 'experiments' / (name + '.py')).exists())

    def test_runtime_and_remaining_tools_do_not_import_retired_probes(self):
        sources = list((POC / 'agent').rglob('*.py')) + list(POC.glob('*.py'))
        sources += list((POC / 'experiments').glob('*.py'))
        violations = []
        for path in sources:
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
                modules = ([node.module or ''] if isinstance(node, ast.ImportFrom) else
                    [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                for module in modules:
                    if set(module.split('.')) & RETIRED:
                        violations.append(f'{path.name}:{node.lineno}:{module}')
        self.assertEqual([], violations)

    def test_current_regressions_and_useful_tools_are_preserved(self):
        for relative in (
            'contract_tests/protocol/test_flat_observation_contract.py', 'contract_tests/observation/test_point_scene_projection.py',
            'contract_tests/retirement/test_approved_restriction_choices.py', 'contract_tests/protocol/test_post_action_authority.py',
            'contract_tests/input/test_adb_only_text_contract.py', 'eval_qwen_visual_decision.py',
            'eval_task_sequences.py', 'capture_click_burst.py', 'run_xy_calibration.py',
            'experiments/audit_runtime_restrictions.py',
            'experiments/probe_simple_input_contract.py', 'experiments/probe_explicit_input_focus.py',
        ):
            with self.subTest(path=relative):
                self.assertTrue((POC / relative).is_file())


if __name__ == '__main__':
    unittest.main()
