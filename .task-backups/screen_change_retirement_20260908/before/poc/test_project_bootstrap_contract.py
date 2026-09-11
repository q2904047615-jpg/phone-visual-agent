import json
import unittest
from pathlib import Path


POC_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = POC_ROOT.parent


class ProjectBootstrapContractTests(unittest.TestCase):
    def test_frontend_browser_dependency_is_declared(self) -> None:
        manifest = json.loads((POC_ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual("1.62.1", manifest["devDependencies"]["playwright"])
        self.assertIn("test:browser", manifest["scripts"])
        self.assertIn("test_frontend_browser_contract.js", manifest["scripts"]["test:browser"])
        self.assertIn("test_phase2_console_browser_contract.js", manifest["scripts"]["test:browser"])

    def test_real_launcher_uses_typed_bootstrap_client(self) -> None:
        launcher = (PROJECT_ROOT / "启动机械臂网页控制台.cmd").read_text(encoding="utf-8")

        self.assertIn("agent_api_cli.py", launcher)
        self.assertIn("bootstrap", launcher)
        self.assertNotIn("Invoke-WebRequest", launcher)
        self.assertNotIn("/api/device", launcher)


if __name__ == "__main__":
    unittest.main()
