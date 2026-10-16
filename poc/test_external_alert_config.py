from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from evals.check_external_alert_config import EXTERNAL_ALERT_PROTOCOL, evaluate_external_alert_config


class ExternalAlertConfigTests(unittest.TestCase):
    def test_missing_credentials_do_not_claim_delivery_readiness(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            report = evaluate_external_alert_config()
        self.assertEqual(EXTERNAL_ALERT_PROTOCOL, report.protocol_version)
        self.assertEqual("not_configured", report.status)
        self.assertEqual("optional_not_configured", report.external_delivery)
        self.assertFalse(report.secret_values_emitted)

    def test_smtp_configuration_is_detected_without_echoing_secret(self) -> None:
        with patch.dict(os.environ, {
            "GMAIL_SENDER": "alerts@example.com",
            "GMAIL_APP_PASSWORD": "fixture-password",
        }, clear=True):
            report = evaluate_external_alert_config()
        self.assertEqual("configured", report.status)
        self.assertEqual("smtp_app_password", report.external_delivery)
        self.assertNotIn("fixture-password", json.dumps(report.as_dict(), ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
