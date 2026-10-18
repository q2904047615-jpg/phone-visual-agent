import tempfile
import unittest
from pathlib import Path

from evals.check_multi_worker_deployment import evaluate


class MultiWorkerDeploymentPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.state = Path(self.temp.name)
        self.base = {
            "PHONE_AGENT_STATE_DIR": str(self.state),
            "ROBOT_API_HOST": "127.0.0.1",
            "ROBOT_API_PORT": "8767",
            "PHONE_AGENT_EXECUTOR_URL": "http://127.0.0.1:8767",
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_owner_requires_explicit_owner_mode(self):
        blocked = evaluate(self.base, role="owner")
        self.assertEqual("blocked", blocked.status)
        ready = evaluate({**self.base, "PHONE_AGENT_EXECUTOR_MODE": "1"}, role="owner")
        self.assertEqual("pass", ready.status)

    def test_ingress_rejects_runtime_owner_mode_and_same_port(self):
        blocked = evaluate({**self.base, "PHONE_AGENT_EXECUTOR_MODE": "1"}, role="ingress")
        self.assertEqual("blocked", blocked.status)
        same_port = evaluate({**self.base, "PHONE_AGENT_EXECUTOR_URL": "http://127.0.0.1:8767"}, role="ingress")
        self.assertEqual("blocked", same_port.status)
        self.assertTrue(any(item.check_id == "port_separation" and item.status == "blocked"
                            for item in same_port.checks))

    def test_runtime_state_requires_dispatch_store_after_initialization(self):
        (self.state / "runtime.sqlite3").touch()
        report = evaluate({**self.base, "PHONE_AGENT_EXECUTOR_MODE": "1"}, role="owner")
        self.assertEqual("blocked", report.status)
        self.assertTrue(any(item.check_id == "shared_dispatch_store" for item in report.checks))

    def test_bad_executor_url_does_not_accept_remote_or_path(self):
        report = evaluate({**self.base, "PHONE_AGENT_EXECUTOR_URL": "http://10.0.0.2:8767/api"}, role="ingress")
        self.assertEqual("blocked", report.status)
        self.assertTrue(any(item.check_id == "executor_url" for item in report.checks))
