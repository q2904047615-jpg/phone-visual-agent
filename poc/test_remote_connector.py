from __future__ import annotations

import json
import secrets
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import httpx
from PIL import Image

from agent.infrastructure.connector_journal import ConnectorJournal
from agent.infrastructure.device_controller_registry import DeviceControllerRegistry
from agent.infrastructure.device_exclusivity import InterProcessLease
from agent.infrastructure.orientation_safety import _mint_single_step_scene_credential, OrientationSafetyError
from agent.infrastructure.remote_connector import (
    ConnectorCommandStore, ConnectorProtocolError, PROTOCOL, RemoteRobotController,
    result_payload, sign_message, validate_command, verify_message,
)
from agent.infrastructure.robot_controller import RobotController
from agent.infrastructure.remote_connector_worker import DeviceConnector

CONNECTOR = "windows-local-01"
DEVICE = "device-local-01"


def command(operation="select_machine_position", args=None, **extra):
    return dict(protocol=PROTOCOL, connector_id=CONNECTOR, device_id=DEVICE,
                command_id=secrets.token_hex(16), operation=operation,
                args={"machine_position": 2} if args is None else args,
                authority=None, expires_at=time.time() + 30, **extra)


class RemoteConnectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.store = ConnectorCommandStore(self.path / "cloud.sqlite3", secret="fixture-secret")

    def enqueue(self, operation="select_machine_position", args=None):
        return self.store.enqueue(connector_id=CONNECTOR, device_id=DEVICE, operation=operation,
                                  args={"machine_position": 2} if args is None else args)

    def result(self, cid, **extra):
        return result_payload(command_id=cid, connector_id=CONNECTOR, device_id=DEVICE, ok=True, value=extra or None)

    def test_no_database_created_for_unused_local_mode(self):
        self.assertFalse(self.store.path.exists())

    def test_signature_binds_identity_nonce_query_method_and_body(self):
        now, nonce = str(time.time()), secrets.token_hex(16)
        target = "/poll?device_id=" + DEVICE
        sig = sign_message("fixture-secret", "GET", target, now, b"", connector_id=CONNECTOR, nonce=nonce)
        verify_message("fixture-secret", "GET", target, now, b"", sig, connector_id=CONNECTOR, nonce=nonce)
        variants = [
            ("GET", target + "2", b"", CONNECTOR, nonce),
            ("POST", target, b"", CONNECTOR, nonce),
            ("GET", target, b"changed", CONNECTOR, nonce),
            ("GET", target, b"", "another-connector", nonce),
            ("GET", target, b"", CONNECTOR, secrets.token_hex(16)),
        ]
        for method, path, body, identity, other_nonce in variants:
            with self.subTest(method=method, path=path, identity=identity), self.assertRaises(ConnectorProtocolError):
                verify_message("fixture-secret", method, path, now, body, sig, connector_id=identity, nonce=other_nonce)

    def test_nan_timestamp_and_persisted_replay_are_rejected(self):
        nonce, target = secrets.token_hex(16), "/poll"
        for now in ("nan", "inf", "-inf"):
            sig = sign_message("fixture-secret", "GET", target, now, b"", connector_id=CONNECTOR, nonce=nonce)
            with self.assertRaises(ConnectorProtocolError):
                verify_message("fixture-secret", "GET", target, now, b"", sig, connector_id=CONNECTOR, nonce=nonce)
        now = str(time.time())
        request = dict(connector_id=CONNECTOR, method="GET", target=target, timestamp=now,
                       nonce=nonce, body=b"", signature=sign_message("fixture-secret", "GET", target, now, b"", connector_id=CONNECTOR, nonce=nonce))
        self.store.authenticate(**request)
        restarted = ConnectorCommandStore(self.store.path, secret="fixture-secret")
        with self.assertRaises(ConnectorProtocolError):
            restarted.authenticate(**request)

    def test_strict_schema_and_foreign_device(self):
        good = command()
        self.assertEqual(validate_command(good, connector_id=CONNECTOR, device_id=DEVICE), good)
        for changed in (
            {"operation": "eval"}, {"args": {"machine_position": True}},
            {"args": {"machine_position": "2"}}, {"args": {"machine_position": 11}},
            {"args": {"machine_position": 2, "shell": "bad"}},
            {"expires_at": float("nan")}, {"expires_at": 0}, {"device_id": "foreign"},
        ):
            with self.subTest(changed=changed), self.assertRaises(ConnectorProtocolError):
                validate_command({**good, **changed}, connector_id=CONNECTOR, device_id=DEVICE)

    def test_claim_is_bound_to_connector_and_device_and_once(self):
        cid = self.enqueue()
        self.assertIsNone(self.store.claim(connector_id="other", device_id=DEVICE))
        self.assertIsNone(self.store.claim(connector_id=CONNECTOR, device_id="other"))
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(lambda _: self.store.claim(connector_id=CONNECTOR, device_id=DEVICE), range(2)))
        self.assertEqual(sum(r is not None for r in results), 1)
        self.assertEqual(next(r for r in results if r)["command_id"], cid)

    def test_result_binding_and_exact_idempotency(self):
        cid = self.enqueue()
        self.store.claim(connector_id=CONNECTOR, device_id=DEVICE)
        payload = self.result(cid, machine_position=2)
        self.assertFalse(self.store.complete(connector_id=CONNECTOR, result={**payload, "device_id": "other"}))
        with self.assertRaises(ConnectorProtocolError):
            self.store.complete(connector_id="other", result=payload)
        self.assertTrue(self.store.complete(connector_id=CONNECTOR, result=payload))
        self.assertTrue(self.store.complete(connector_id=CONNECTOR, result=payload))
        self.assertFalse(self.store.complete(connector_id=CONNECTOR, result={**payload, "value": "conflicting"}))
        self.assertEqual(self.store.wait(cid), payload)

    def test_queued_timeout_is_not_dispatched_claimed_is_unknown(self):
        for claimed, status in ((False, "not_dispatched"), (True, "unknown")):
            cid = self.enqueue()
            if claimed:
                self.store.claim(connector_id=CONNECTOR, device_id=DEVICE)
            with self.assertRaises(ConnectorProtocolError):
                self.store.wait(cid, timeout=0)
            with self.store.connect() as db:
                self.assertEqual(db.execute("SELECT status FROM connector_commands_v2 WHERE command_id=?", (cid,)).fetchone()[0], status)
            self.assertIsNone(self.store.claim(connector_id=CONNECTOR, device_id=DEVICE))

    def test_late_result_is_evidence_only_never_success(self):
        cid = self.enqueue()
        self.store.claim(connector_id=CONNECTOR, device_id=DEVICE)
        with self.assertRaises(ConnectorProtocolError):
            self.store.wait(cid, timeout=0)
        payload = self.result(cid)
        self.assertTrue(self.store.complete(connector_id=CONNECTOR, result=payload))
        self.assertTrue(self.store.complete(connector_id=CONNECTOR, result=payload))
        with self.assertRaises(ConnectorProtocolError):
            self.store.wait(cid)

    def test_registry_preserves_position_and_shared_controller_locks(self):
        registry = self.path / "registry.json"
        registry.write_text(json.dumps({"version": 1, "default_device_id": DEVICE, "devices": [
            {"enabled": True, "device_id": DEVICE, "connector_id": CONNECTOR, "machine_position": 2},
            {"enabled": True, "device_id": "device-local-02", "connector_id": CONNECTOR, "machine_position": 3},
        ]}), encoding="utf-8")
        result = DeviceControllerRegistry(registry, promotable_actions=(), connector_store=self.store)
        first, second = result.controller(DEVICE), result.controller("device-local-02")
        self.assertEqual(first.machine_position, 2)
        self.assertIs(first.operation_lock, second.operation_lock)
        self.assertIs(first.capture_lock, second.capture_lock)
        self.assertTrue(first.hardware_capabilities()["swipe"])
        self.assertEqual(result.descriptors()[0]["machine_position"], 2)

    def test_status_remains_readable_on_disconnect(self):
        proxy = RemoteRobotController(device_id=DEVICE, connector_id=CONNECTOR, store=self.store, machine_position=2)
        with patch.object(self.store, "wait", side_effect=ConnectorProtocolError("offline")):
            self.assertFalse(proxy.device_status()["controller_online"])


class FakeHardware(RobotController):
    """Exercises the actual physical gate without Windows, models or devices."""
    def __init__(self):
        super().__init__(device_id=DEVICE)
        self.frame = Image.new("RGB", (32, 64), (34, 68, 102))
        self.calls = []

    def vision_capture(self):
        return self.frame.copy()

    def capture_preview(self, quality=72):
        output = BytesIO()
        self.frame.save(output, format="JPEG", quality=quality)
        return output.getvalue()

    def device_status(self):
        return {"controller_online": True, "camera_online": True, "busy": False}

    def select_machine_position(self, machine_position):
        self.calls.append(("position", machine_position))
        self.machine_position = machine_position

    def vision_tap_relative(self, x, y):
        self._consume_physical_execution("tap_semantic", self.frame)
        self.calls.append(("tap", x, y))
        self._last_click_receipt = {"transport_ack": "input_events_dispatched", "x": x}
        return x, y

    def vision_android_home(self):
        self._consume_physical_execution("home", self.frame)
        self.calls.append(("home",))
        return (0, 1)


class ConnectorWorkerTests(unittest.TestCase):
    enqueue = RemoteConnectorTests.enqueue

    def setUp(self):
        RemoteConnectorTests.setUp(self)
        self.hardware = FakeHardware()
        self.reject_next_result = False
        self.client = httpx.Client(transport=httpx.MockTransport(self.handle))
        self.worker = DeviceConnector(base_url="https://connector.test", connector_id=CONNECTOR,
            secret="fixture-secret", registry_path=self.path / "unused", device_id=DEVICE,
            journal_path=self.path / "journal.sqlite3", controller=self.hardware, client=self.client,
            lease_directory=self.path / "leases")
        self.worker.__enter__()
        self.addCleanup(self.worker.__exit__, None, None, None)
        self.proxy = RemoteRobotController(device_id=DEVICE, connector_id=CONNECTOR, store=self.store, machine_position=1)

    def handle(self, request):
        self.store.authenticate(connector_id=request.headers["X-Connector-Id"], method=request.method,
            target=request.url.raw_path.decode(), timestamp=request.headers["X-Connector-Timestamp"],
            nonce=request.headers["X-Connector-Nonce"], body=request.content, signature=request.headers["X-Connector-Signature"])
        if request.method == "GET":
            cmd = self.store.claim(connector_id=CONNECTOR, device_id=request.url.params["device_id"])
            return httpx.Response(200, json=cmd) if cmd else httpx.Response(204)
        value = json.loads(request.content)
        accepted = self.store.complete(connector_id=CONNECTOR, result=value)
        if self.reject_next_result:
            self.reject_next_result = False
            raise httpx.ReadError("lost acknowledgement")
        return httpx.Response(200 if accepted else 409, json={"accepted": accepted, "command_id": value["command_id"]})

    def roundtrip(self, function, *args):
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(function, *args)
            deadline = time.monotonic() + 5
            while not future.done() and time.monotonic() < deadline:
                self.worker.run_once()
                time.sleep(.005)
            return future.result(timeout=1)

    def test_position_2_then_10_preview_and_status_roundtrip(self):
        self.roundtrip(self.proxy.select_machine_position, 2)
        self.roundtrip(self.proxy.select_machine_position, 10)
        self.assertEqual(self.proxy.machine_position, 10)
        self.assertEqual(self.hardware.machine_position, 10)
        self.assertTrue(self.roundtrip(self.proxy.device_status)["camera_online"])
        self.assertTrue(self.roundtrip(self.proxy.capture_preview).startswith(b"\xff\xd8"))
        self.assertEqual(self.hardware.calls, [("position", 2), ("position", 10)])

    def test_lossless_observation_one_action_and_receipt(self):
        frame = self.roundtrip(self.proxy.vision_capture)
        self.assertEqual(frame.tobytes(), self.hardware.frame.tobytes())
        credential = _mint_single_step_scene_credential(device_id=DEVICE, scene_fingerprint="scene", frame=frame)
        self.proxy.arm_physical_execution(credential, action="tap_semantic", scene_fingerprint="scene")
        self.assertEqual(self.roundtrip(self.proxy.vision_tap_relative, 321, 654), [321, 654])
        self.assertEqual(self.proxy.consume_last_click_receipt()["x"], 321)
        with self.assertRaises((ConnectorProtocolError, OrientationSafetyError)):
            self.proxy.vision_tap_relative(321, 654)
        self.assertEqual(self.hardware.calls, [("tap", 321, 654)])

    def test_home_variation_and_changed_canvas_rejected_by_original_gate(self):
        for change in (False, True):
            frame = self.roundtrip(self.proxy.vision_capture)
            credential = _mint_single_step_scene_credential(device_id=DEVICE, scene_fingerprint="scene", frame=frame)
            self.proxy.arm_physical_execution(credential, action="home", scene_fingerprint="scene")
            if change:
                self.hardware.frame = Image.new("RGB", (64, 32))
                with self.assertRaises(ConnectorProtocolError):
                    self.roundtrip(self.proxy.vision_android_home)
            else:
                self.roundtrip(self.proxy.vision_android_home)
        self.assertEqual(self.hardware.calls, [("home",)])

    def test_lost_ack_retries_result_without_reexecuting(self):
        cid = self.enqueue()
        self.reject_next_result = True
        with self.assertRaises(httpx.ReadError):
            self.worker.run_once()
        self.assertTrue(self.store.wait(cid)["ok"])
        self.worker.run_once()
        self.assertEqual(self.hardware.calls, [("position", 2)])
        self.assertEqual(self.worker.journal.pending(), [])

    def test_pixel_motion_is_not_a_new_action_veto(self):
        frame = self.roundtrip(self.proxy.vision_capture)
        credential = _mint_single_step_scene_credential(device_id=DEVICE, scene_fingerprint="scene", frame=frame)
        self.proxy.arm_physical_execution(credential, action="tap_semantic", scene_fingerprint="scene")
        self.hardware.frame = Image.new("RGB", frame.size, (102, 68, 34))
        self.roundtrip(self.proxy.vision_tap_relative, 300, 600)
        self.assertEqual(self.hardware.calls, [("tap", 300, 600)])

    def test_transport_cannot_reuse_capture_under_a_new_command_id(self):
        frame = self.roundtrip(self.proxy.vision_capture)
        credential = _mint_single_step_scene_credential(device_id=DEVICE, scene_fingerprint="scene", frame=frame)
        authority = {"capture_id": self.proxy._remote_capture_id, "credential": credential.to_dict()}
        first = {**command("vision_tap_relative", {"x": 300, "y": 600}), "authority": authority}
        self.assertTrue(self.worker.execute(first)["ok"])
        second = {**first, "command_id": secrets.token_hex(16)}
        self.assertFalse(self.worker.execute(second)["ok"])
        self.assertEqual(self.hardware.calls, [("tap", 300, 600)])

    def test_started_delivery_restart_never_reexecutes(self):
        cmd = command()
        self.worker.journal.begin(cmd)
        reopened = ConnectorJournal(self.worker.journal.path)
        recovered = reopened.pending()
        self.assertIn("unknown", recovered[0]["error"])
        result = self.worker.execute(cmd)
        self.assertFalse(result["ok"])
        self.assertEqual(self.hardware.calls, [])

    def test_duplicate_id_conflict_and_exact_duplicate(self):
        cmd = command()
        first = self.worker.execute(cmd)
        self.assertEqual(self.worker.execute(cmd), first)
        with self.assertRaises(ConnectorProtocolError):
            self.worker.execute({**cmd, "args": {"machine_position": 3}})
        self.assertEqual(self.hardware.calls, [("position", 2)])

    def test_expired_foreign_and_arbitrary_commands_do_not_execute(self):
        for changed in ({"expires_at": 0}, {"device_id": "foreign"}, {"operation": "request_secret"}):
            with self.subTest(changed=changed), self.assertRaises(ConnectorProtocolError):
                self.worker.execute({**command(), **changed})
        self.assertEqual(self.hardware.calls, [])

    def test_https_required_and_local_lease_excludes_other_api(self):
        with self.assertRaises(ConnectorProtocolError):
            DeviceConnector(base_url="http://connector.test", connector_id=CONNECTOR, secret="s",
                registry_path=self.path / "unused", device_id=DEVICE, journal_path=self.path / "unused-db")
        lease = InterProcessLease(self.path / "leases" / "physical_hardware_action.lease", owner_id="other-api", metadata={})
        self.assertFalse(lease.acquire())


if __name__ == "__main__":
    unittest.main()
