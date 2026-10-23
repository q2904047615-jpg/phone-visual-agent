"""Real TCP: two stateless API processes, one Runtime and a simulated Windows device."""
import json
import multiprocessing
import socket
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from agent.infrastructure.remote_connector import ConnectorCommandStore
from agent.infrastructure.remote_connector import RemoteRobotController
from agent.infrastructure.remote_device_tools import RemoteTextTransport, RemoteAppLauncher, describe_remote_tools
from agent.infrastructure.remote_connector_worker import DeviceConnector
from local_agent_api_client import LocalAgentApiClient, LocalAgentApiError
from test_remote_connector import FakeHardware
from test_support.multi_worker_server import run_server


class LoopbackTransport(httpx.BaseTransport):
    """Test-only HTTPS termination substitute, restricted to fixture ports."""
    def __init__(self, ports):
        self.ports = ports
        self.index = 0
        self.inner = httpx.HTTPTransport(retries=0)
        self.drop_next_result_ack = False

    def handle_request(self, request):
        port = self.ports[self.index % len(self.ports)]
        self.index += 1
        request.url = request.url.copy_with(scheme="http", host="127.0.0.1", port=port)
        request.headers["host"] = f"127.0.0.1:{port}"
        response = self.inner.handle_request(request)
        if self.drop_next_result_ack and request.url.path.endswith("/device-connector/v1/result"):
            self.drop_next_result_ack = False
            response.read()
            response.close()
            raise httpx.ReadError("fixture dropped the committed acknowledgement")
        return response

    def close(self):
        self.inner.close()


class RemoteConnectorHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="connector-http-")
        cls.directory = Path(cls.tmp.name)
        cls.processes, cls.sockets = [], []
        cls.addClassCleanup(cls.cleanup)
        context = multiprocessing.get_context("spawn")
        ports = []
        for role in ("executor", "api", "api"):
            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
            listener.listen(128)
            cls.sockets.append(listener)
            port = listener.getsockname()[1]
            ports.append(port)
            parent, child = context.Pipe()
            process = context.Process(target=run_server,
                args=(role, str(cls.directory), listener, ports[0], child, True))
            process.start()
            cls.processes.append(process)
            child.close()
            if not parent.poll(25):
                raise AssertionError(f"fixture {role} failed to start: {process.exitcode}")
            metadata = parent.recv()
            parent.close()
            if role == "executor":
                cls.identities = metadata["identities"]
            elif metadata["runtime_imported"]:
                raise AssertionError("API worker imported Runtime")
        cls.ports = ports[1:]
        deadline = time.monotonic() + 20
        while True:
            try:
                with cls.api() as client:
                    client.bootstrap()
                break
            except LocalAgentApiError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.1)

    @classmethod
    def cleanup(cls):
        for process in reversed(cls.processes):
            if process.is_alive():
                process.terminate()
            process.join(5)
        for listener in cls.sockets:
            listener.close()
        cls.tmp.cleanup()

    @classmethod
    def api(cls, user="local-owner"):
        return LocalAgentApiClient(transport=LoopbackTransport(cls.ports), timeout_seconds=15,
            principal_token=cls.identities[user]["principal_token"])

    def test_cross_worker_station_preview_permission_and_lost_ack(self):
        hardware = FakeHardware()
        transport = LoopbackTransport(self.ports)
        client = httpx.Client(transport=transport, timeout=10)
        stop = threading.Event()
        errors = []
        # Reuse only the isolated registry/ADB fixtures; all delivery below uses
        # real TCP through the two ingress processes and the Runtime process.
        from test_remote_device_tools import RemoteDeviceToolsTests
        tools_fixture = RemoteDeviceToolsTests()
        tools_fixture.setUp()
        self.addCleanup(tools_fixture.doCleanups)
        worker = DeviceConnector(base_url="https://connector.test", connector_id="windows-local-01",
            secret="isolated-connector-secret", device_id="device-local-01", controller=hardware,
            registry_path=self.directory / "unused", journal_path=self.directory / "windows.sqlite3",
            lease_directory=self.directory / "windows-leases", client=client)
        worker.tools = tools_fixture.worker.tools

        def run_connector():
            with worker:
                while not stop.is_set():
                    try:
                        worker.run_once()
                    except httpx.TransportError:
                        pass  # Next iteration resubmits only the durable result.
                    except Exception as exc:
                        errors.append(repr(exc))
                        break
                    stop.wait(.02)

        thread = threading.Thread(target=run_connector)
        thread.start()
        try:
            with self.api() as api:
                transport.drop_next_result_ack = True
                response = api.select_machine_position("device-local-01", 2)
                self.assertEqual(response["machine_position"], 2)
                self.assertTrue(response["status"]["controller_online"])
                self.assertEqual(api.select_machine_position("device-local-01", 10)["machine_position"], 10)
                self.assertEqual(hardware.calls, [("position", 2), ("position", 10)])
                with self.api("worker-other") as other:
                    with self.assertRaises(LocalAgentApiError) as denial:
                        other.select_machine_position("device-local-01", 3)
                    self.assertEqual(denial.exception.details.status_code, 403)
                self.assertEqual(hardware.calls, [("position", 2), ("position", 10)])
            # Exercise the very same HTTPS connector poll/result through both
            # ingresses, including PNG payloads in dispatch-ticket POSTs.
            store = ConnectorCommandStore(self.directory / "connector.sqlite3", secret="isolated-connector-secret")
            cid = store.enqueue(connector_id="windows-local-01", device_id="device-local-01",
                                operation="vision_capture", args={})
            result = store.wait(cid, timeout=10)
            self.assertTrue(result["ok"])
            self.assertEqual(result["value"]["capture_id"], cid)
            proxy = RemoteRobotController(device_id="device-local-01", connector_id="windows-local-01", store=store, machine_position=10)
            proxy.select_machine_position(2)
            description = describe_remote_tools(proxy)
            text = RemoteTextTransport(proxy, description)
            launcher = RemoteAppLauncher(proxy, description)
            for value in ("跨入口中文 🌍", "\n", ""):
                proxy.vision_capture()
                scope = tools_fixture.scope(text, value)
                receipt = text.append_text(scope, value) if value else text.clear_text(scope)
                self.assertTrue(receipt.accepted)
            proxy.vision_capture()
            launcher.launch(launcher.resolve("", "测试应用").launch_ref)
            self.assertEqual(len(tools_fixture.broadcasts()), 3)
            self.assertTrue(all(call[2] == "serial-2" for call in tools_fixture.calls))
            self.assertEqual(sum("monkey" in call for call in tools_fixture.calls), 1)
        finally:
            stop.set()
            thread.join(12)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
