from __future__ import annotations

import json
import secrets
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import web_app
from agent.infrastructure.remote_connector import ConnectorCommandStore, result_payload, sign_message

CID = "windows-local-01"
DEVICE = "device-local-01"
POLL = "/api/device-connector/v1/poll?device_id=" + DEVICE
RESULT = "/api/device-connector/v1/result"


class RemoteConnectorWebTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = ConnectorCommandStore(Path(self.tmp.name) / "connector.sqlite3", secret="fixture-secret")
        self.addCleanup(patch.stopall)
        patch.object(web_app.runtime, "connector_store", self.store).start()
        patch.object(web_app.runtime.device_controllers, "descriptors", return_value=[
            {"device_id": DEVICE, "connector_id": CID},
            {"device_id": "other-device", "connector_id": "other-connector"},
        ]).start()
        self.client = TestClient(web_app.app)
        self.addCleanup(self.client.close)

    def headers(self, method, target, body=b"", cid=CID):
        timestamp, nonce = str(time.time()), secrets.token_hex(16)
        return {"X-Connector-Id": cid, "X-Connector-Timestamp": timestamp, "X-Connector-Nonce": nonce,
                "X-Connector-Signature": sign_message("fixture-secret", method, target, timestamp, body, connector_id=cid, nonce=nonce),
                "Content-Type": "application/json"}

    def enqueue(self):
        return self.store.enqueue(connector_id=CID, device_id=DEVICE, operation="device_status", args={})

    def post(self, payload, cid=CID):
        body = json.dumps(payload).encode()
        return self.client.post(RESULT, content=body, headers=self.headers("POST", RESULT, body, cid))

    def test_signed_poll_result_and_exact_idempotency(self):
        command_id = self.enqueue()
        response = self.client.get(POLL, headers=self.headers("GET", POLL))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["command_id"], command_id)
        payload = result_payload(command_id=command_id, connector_id=CID, device_id=DEVICE, ok=True, value={"controller_online": True})
        self.assertEqual(self.post(payload).status_code, 200)
        self.assertEqual(self.post(payload).status_code, 200)
        self.assertEqual(self.post({**payload, "value": False}).status_code, 409)
        self.assertTrue(self.store.wait(command_id)["ok"])

    def test_missing_signature_and_replay_do_not_claim_next_command(self):
        self.assertEqual(self.client.get(POLL).status_code, 403)  # unknown identity has no device binding
        first, second = self.enqueue(), self.enqueue()
        headers = self.headers("GET", POLL)
        self.assertEqual(self.client.get(POLL, headers=headers).json()["command_id"], first)
        self.assertEqual(self.client.get(POLL, headers=headers).status_code, 401)
        self.assertEqual(self.client.get(POLL, headers=self.headers("GET", POLL)).json()["command_id"], second)

    def test_other_registered_connector_cannot_impersonate_or_submit(self):
        cid = self.enqueue()
        self.client.get(POLL, headers=self.headers("GET", POLL))
        other = "/api/device-connector/v1/poll?device_id=other-device"
        self.assertEqual(self.client.get(other, headers=self.headers("GET", other, cid="other-connector")).status_code, 401)
        payload = result_payload(command_id=cid, connector_id="other-connector", device_id="other-device", ok=True)
        self.assertEqual(self.post(payload, cid="other-connector").status_code, 401)
        with self.store.connect() as db:
            self.assertEqual(db.execute("SELECT status FROM connector_commands_v2 WHERE command_id=?", (cid,)).fetchone()[0], "claimed")

    def test_signed_result_still_requires_device_binding(self):
        cid = self.enqueue()
        self.client.get(POLL, headers=self.headers("GET", POLL))
        payload = result_payload(command_id=cid, connector_id=CID, device_id="other-device", ok=True)
        self.assertEqual(self.post(payload).status_code, 403)
        self.assertEqual(self.post({**payload, "device_id": DEVICE, "ok": "true"}).status_code, 422)

    def test_query_tampering_is_rejected(self):
        headers = self.headers("GET", POLL + "&ignored=1")
        self.assertEqual(self.client.get(POLL, headers=headers).status_code, 401)


if __name__ == "__main__":
    unittest.main()
