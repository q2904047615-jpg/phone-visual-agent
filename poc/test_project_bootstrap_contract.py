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

    def test_both_launchers_check_only_current_web_dependencies(self) -> None:
        for name in ("启动机械臂网页控制台.cmd", "启动网页控制台_模拟模式.cmd"):
            with self.subTest(launcher=name):
                launcher = (PROJECT_ROOT / name).read_text(encoding="utf-8")
                self.assertIn('import fastapi, uvicorn, httpx"', launcher)
                self.assertNotIn("pypinyin", launcher)

    def test_retired_deepseek_configuration_launchers_are_absent(self) -> None:
        for name in ("configure_deepseek_key.ps1", "配置DeepSeek文本理解密钥.ps1", "配置DeepSeek文本理解密钥.cmd"):
            with self.subTest(launcher=name):
                self.assertFalse((PROJECT_ROOT / name).exists())


if __name__ == "__main__":
    unittest.main()
