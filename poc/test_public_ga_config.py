from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evals.check_public_ga_config import (
    GITHUB_CALLBACK_PATH,
    PUBLIC_GA_CONFIG_PROTOCOL,
    evaluate_public_ga_config,
)


class PublicGaConfigTests(unittest.TestCase):
    def _config(self, root: Path) -> dict[str, str]:
        mapping = root / "identity-map.json"
        mapping.write_text(json.dumps({"42": {"organization_id": "org", "user_id": "owner"}}), encoding="utf-8")
        lease = root / "lease"
        lease.mkdir()
        return {
            "PUBLIC_BASE_URL": "https://agent.example.com",
            "GITHUB_OAUTH_CLIENT_ID": "public-client-id",
            "GITHUB_OAUTH_CLIENT_SECRET": "[REDACTED_SECRET]",
            "GITHUB_OAUTH_IDENTITY_MAP_FILE": str(mapping),
            "GITHUB_OAUTH_REDIRECT_URI": f"https://agent.example.com{GITHUB_CALLBACK_PATH}",
            "DASHSCOPE_API_KEY": "[REDACTED_SECRET]",
            "VISION_MODEL": "qwen3-vl-plus",
            "VISION_MODEL_BASE_URL": "https://dashscope.example.com/compatible-mode/v1",
            "PHONE_VISUAL_AGENT_LEASE_DIR": str(lease),
        }

    def test_complete_public_configuration_passes_without_echoing_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = evaluate_public_ga_config(self._config(Path(temporary)))
        self.assertEqual(PUBLIC_GA_CONFIG_PROTOCOL, report.protocol_version)
        self.assertEqual("pass", report.status)
        encoded = json.dumps(report.as_dict(), ensure_ascii=False)
        self.assertNotIn("[REDACTED_SECRET]", encoded)

    def test_missing_secret_is_blocked_and_marked_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            values = self._config(Path(temporary))
            values.pop("DASHSCOPE_API_KEY")
            report = evaluate_public_ga_config(values)
        self.assertEqual("blocked", report.status)
        check = next(item for item in report.checks if item.check_id == "required_secrets")
        self.assertTrue(check.sensitive)
        self.assertIn("DASHSCOPE_API_KEY", check.detail)

    def test_local_http_and_cross_origin_callback_are_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            values = self._config(Path(temporary))
            values["PUBLIC_BASE_URL"] = "http://127.0.0.1:8765"
            values["GITHUB_OAUTH_REDIRECT_URI"] = f"https://other.example.com{GITHUB_CALLBACK_PATH}"
            report = evaluate_public_ga_config(values)
        self.assertEqual("blocked", report.status)
        self.assertTrue(any(item.check_id == "public_base_url" and item.status == "blocked" for item in report.checks))
        self.assertTrue(any(item.check_id == "github_oauth_redirect" and item.status == "blocked" for item in report.checks))

    def test_invalid_identity_map_and_relative_lease_dir_are_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            values = self._config(Path(temporary))
            values["GITHUB_OAUTH_IDENTITY_MAP_FILE"] = str(Path(temporary) / "missing.json")
            values["PHONE_VISUAL_AGENT_LEASE_DIR"] = "lease"
            report = evaluate_public_ga_config(values)
        self.assertEqual("blocked", report.status)
        self.assertTrue(any(item.check_id == "github_oauth_identity_map" and item.status == "blocked" for item in report.checks))
        self.assertTrue(any(item.check_id == "runtime_lease_dir" and item.status == "blocked" for item in report.checks))

    def test_model_and_lease_defaults_are_derived(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            values = self._config(Path(temporary))
            values.pop("VISION_MODEL")
            values.pop("VISION_MODEL_BASE_URL")
            values.pop("PHONE_VISUAL_AGENT_LEASE_DIR")
            report = evaluate_public_ga_config(values)
        self.assertEqual("pass", report.status)
        self.assertEqual("derived", next(item.status for item in report.checks if item.check_id == "vision_model"))
        self.assertEqual("derived", next(item.status for item in report.checks if item.check_id == "vision_model_base_url"))
        self.assertEqual("derived", next(item.status for item in report.checks if item.check_id == "runtime_lease_dir"))


if __name__ == "__main__":
    unittest.main()
