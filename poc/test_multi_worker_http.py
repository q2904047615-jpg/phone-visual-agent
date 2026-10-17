"""Real TCP/ASGI processes with the production API and simulated adapters."""
import json
import os
import multiprocessing
import socket
import sqlite3
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
from local_agent_api_client import LocalAgentApiClient, LocalAgentApiError
from test_support.multi_worker_server import run_server, try_second_executor


class WorkerTransport(httpx.BaseTransport):
    """Use the official SDK/OpenAPI contract over real isolated TCP ports."""
    def __init__(self, port):
        self.port = port
        self.inner = httpx.HTTPTransport(retries=0)

    def handle_request(self, request):
        request.url = request.url.copy_with(port=self.port)
        return self.inner.handle_request(request)

    def close(self):
        self.inner.close()


class MultiWorkerHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="agent-workers-")
        cls.directory = Path(cls.tmp.name)
        cls.context = multiprocessing.get_context("spawn")
        cls.processes, cls.sockets = [], []
        cls.addClassCleanup(cls.cleanup)
        cls.executor_socket = cls.listener()
        cls.executor_port = cls.executor_socket.getsockname()[1]
        cls.executor, cls.identity = cls.start_server("executor", cls.executor_socket)
        cls.worker_sockets = [cls.listener(), cls.listener()]
        cls.workers = [cls.start_server("api", sock)[0] for sock in cls.worker_sockets]
        cls.ports = [sock.getsockname()[1] for sock in cls.worker_sockets]
        cls.wait_ready()

    @classmethod
    def listener(cls, port=0):
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", port))
        listener.listen(128)
        cls.sockets.append(listener)
        return listener

    @classmethod
    def start_server(cls, role, listener):
        parent, child = cls.context.Pipe()
        process = cls.context.Process(target=run_server,
            args=(role, str(cls.directory), listener, cls.executor_port, child))
        process.start()
        cls.processes.append(process)
        child.close()
        if not parent.poll(25):
            raise AssertionError(f"{role} did not initialize; exit={process.exitcode}")
        metadata = parent.recv()
        parent.close()
        if role == "api" and metadata["runtime_imported"]:
            raise AssertionError("API worker constructed a Runtime")
        return process, metadata

    @classmethod
    def client(cls, index=0, user="local-owner"):
        token = cls.identity["identities"][user]["principal_token"]
        return LocalAgentApiClient(transport=WorkerTransport(cls.ports[index]),
                                   principal_token=token, timeout_seconds=15)

    @classmethod
    def wait_ready(cls):
        deadline = time.monotonic() + 20
        while True:
            try:
                for index in (0, 1):
                    with cls.client(index) as client:
                        client.bootstrap()
                return
            except LocalAgentApiError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(.1)

    @classmethod
    def stop(cls, process):
        if process.is_alive():
            process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)

    @classmethod
    def cleanup(cls):
        for process in reversed(cls.processes):
            cls.stop(process)
        for sock in cls.sockets:
            sock.close()
        cls.tmp.cleanup()

    def test_a_distinct_workers_share_bootstrap_and_control_plane(self):
        self.assertEqual(3, len({self.executor.pid, *(p.pid for p in self.workers)}))
        with self.client(0) as first, self.client(1) as second:
            self.assertEqual(first.bootstrap(), second.bootstrap())
            self.assertEqual(first._token, second._token)
            self.assertEqual(first.control_plane_organization("local-org")["data"]["organization"],
                             second.control_plane_organization("local-org")["data"]["organization"])

    def test_b_oauth_cross_worker_and_session_revocation(self):
        with self.client(0) as first, self.client(1) as second:
            start = first.control_plane_github_oauth_start()
            state = parse_qs(urlsplit(start["data"]["authorization_url"]).query)["state"][0]
            result = second.control_plane_github_oauth_callback(state=state, code="fixture-code")
            session = result["data"]["session"]
            first._client.cookies.set("agent_principal_v2", session["principal_token"], path="/api")
            second._client.cookies.set("agent_principal_v2", session["principal_token"], path="/api")
            self.assertTrue(first.control_plane_browser_session()["data"]["authenticated"])
            self.assertTrue(second.control_plane_browser_session()["data"]["authenticated"])
            first.control_plane_browser_logout()
            self.assertFalse(second.control_plane_browser_session()["data"]["authenticated"])

    def test_c_idempotency_and_conflicting_content_across_workers(self):
        def create(index, name="Concurrent team"):
            with self.client(index) as client:
                return client.control_plane_create_team("local-org", team_id="parallel-team", name=name,
                    member_ids=[], principal_token=client._principal_token, idempotency_key="same-team")
        with ThreadPoolExecutor(2) as pool:
            replies = list(pool.map(create, (0, 1)))
        self.assertEqual(2, len(replies))
        with self.client(1) as client:
            teams = client.control_plane_teams("local-org")["data"]["teams"]
            self.assertEqual(1, sum(t["team_id"] == "parallel-team" for t in teams))
        with self.assertRaises(LocalAgentApiError) as conflict:
            create(1, "Changed content")
        self.assertEqual(409, conflict.exception.details.status_code)

    def test_d_permission_change_visible_to_other_worker(self):
        with self.client(0) as owner, self.client(1, "worker-admin") as admin:
            admin.control_plane_principals("local-org")
            owner.control_plane_update_principal("local-org", "worker-admin", role="viewer",
                status="active", expected_revision=0, idempotency_key="downgrade-admin")
            with self.assertRaises(LocalAgentApiError) as denied:
                admin.control_plane_principals("local-org")
            self.assertIn(denied.exception.details.status_code, {401, 403})

    def test_e_run_cross_worker_isolation_events_and_cancel(self):
        with self.client(0) as first, self.client(1) as second, self.client(1, "worker-other") as other:
            try:
                started = first.start_session(text="isolated navigation", device_id="device-local-01")
            except LocalAgentApiError as error:
                self.fail(str(error.details.detail))
            session = started["session"]["session_id"]
            self.assertEqual("awaiting_confirmation", second.get_session(session)["session"]["status"])
            with self.assertRaises(LocalAgentApiError) as denied:
                other.get_session(session)
            self.assertEqual(404, denied.exception.details.status_code)
            page = second._request("GET", "/api/agent/generic-supervised/{session_id}/events",
                                   path_params={"session_id":session}, read_only=True)
            self.assertTrue(page["events"])
            repeated = first._request("GET", "/api/agent/generic-supervised/{session_id}/events",
                path_params={"session_id":session}, query_params={"after_sequence":str(page["next_sequence"])}, read_only=True)
            self.assertEqual([], repeated["events"])
            second.cancel_session(session)
            self.assertEqual("cancelled", first.get_session(session)["session"]["status"])

    def test_f_same_device_competition_and_distinct_device(self):
        def start(index):
            with self.client(index) as client:
                try:
                    return client.start_session(text="compete", device_id="device-local-01")
                except LocalAgentApiError as error:
                    return error
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(start, (0, 1)))
        successes = [r for r in results if isinstance(r, dict)]
        self.assertEqual(1, len(successes), str(results))
        self.assertEqual(409, next(r for r in results if isinstance(r, LocalAgentApiError)).details.status_code)
        with self.client(1) as client:
            other = client.start_session(text="other device", device_id="mock-device-2")
            client.cancel_session(other["session"]["session_id"])
            client.cancel_session(successes[0]["session"]["session_id"])

    def test_g_worker_restart_does_not_mark_live_run_recovered(self):
        with self.client(0) as client:
            started = client.start_session(text="worker restart", device_id="device-local-01")
            session = started["session"]["session_id"]
        self.stop(self.workers[1])
        self.worker_sockets[1].close()
        self.worker_sockets[1] = self.listener()
        self.ports[1] = self.worker_sockets[1].getsockname()[1]
        self.workers[1] = self.start_server("api", self.worker_sockets[1])[0]
        self.wait_ready()
        with self.client(1) as client:
            self.assertEqual("awaiting_confirmation", client.get_session(session)["session"]["status"])
            client.confirm_once(session)
            self.assertEqual("succeeded", client.get_session(session)["session"]["status"])

    def test_h_bounded_concurrent_read_write_soak(self):
        rounds = int(os.environ.get("MULTI_WORKER_SOAK_ROUNDS", "100"))
        def roundtrip(number):
            with self.client(number % 2) as client:
                client.control_plane_create_team("local-org", team_id=f"soak-{number}", name="Soak",
                    member_ids=[], principal_token=client._principal_token, idempotency_key=f"soak-{number}")
                client.control_plane_teams("local-org")
            return True
        with ThreadPoolExecutor(4) as pool:
            self.assertTrue(all(pool.map(roundtrip, range(rounds))))
        with closing(sqlite3.connect(self.directory / "dispatch.sqlite3")) as db:
            self.assertEqual(0, db.execute("SELECT COUNT(*) FROM dispatch_tickets WHERE status!='responded'").fetchone()[0])

    def test_i_second_executor_rejected_before_runtime_import(self):
        parent, child = self.context.Pipe()
        process = self.context.Process(target=try_second_executor, args=(str(self.directory), child))
        process.start()
        self.processes.append(process)
        child.close()
        self.assertTrue(parent.poll(10))
        self.assertEqual({"rejected":True,"runtime_imported":False}, parent.recv())
        parent.close()
        process.join(5)
        self.assertEqual(0, process.exitcode)

    def test_j_last_quota_competes_across_distinct_devices(self):
        with self.client() as client:
            quota = next(q for q in client.control_plane_quotas("local-org")["data"]["quotas"] if q["quota_key"] == "daily")
            ceiling = quota["usage"]["runs"] + 1
            client.control_plane_update_quota("local-org", "daily", period="daily", max_runs=ceiling,
                principal_token=client._principal_token, idempotency_key="last-slot")
        def start(index):
            with self.client(index) as client:
                try:
                    return client.start_session(text="quota", device_id=("device-local-01", "mock-device-2")[index])
                except LocalAgentApiError as error:
                    return error
        try:
            with ThreadPoolExecutor(2) as pool:
                results = list(pool.map(start, (0,1)))
            successes = [r for r in results if isinstance(r, dict)]
            self.assertEqual(1,len(successes))
            rejected = next(r for r in results if isinstance(r, LocalAgentApiError))
            self.assertIn("quota_exceeded", str(rejected.details.detail))
            with self.client() as client:
                client.cancel_session(successes[0]["session"]["session_id"])
                after = next(q for q in client.control_plane_quotas("local-org")["data"]["quotas"] if q["quota_key"] == "daily")
                self.assertEqual(0, after["reserved"]["runs"])
        finally:
            with self.client() as client:
                client.control_plane_update_quota("local-org", "daily", period="daily",
                    max_runs=quota["max_runs"] or 10000,
                    max_physical_actions=quota["max_physical_actions"], max_model_tokens=quota["max_model_tokens"],
                    principal_token=client._principal_token, idempotency_key="restore-quota")

    def test_k_async_ticket_cross_worker_completed_result(self):
        with self.client(0) as first, self.client(1) as second:
            started = first._request("POST", "/api/agent/generic-supervised/start-async",
                payload={"text":"async navigation", "device_id":"device-local-01", "auto_advance":True}, read_only=False)
            task_id = started["task_id"]
            deadline = time.monotonic() + 10
            while True:
                task = second._request("GET", "/api/agent/generic-supervised/start-async/{task_id}",
                                       path_params={"task_id":task_id}, read_only=True)
                if task["status"] in {"completed","failed"}:
                    break
                self.assertLess(time.monotonic(), deadline)
                time.sleep(.05)
            self.assertEqual("completed", task["status"], str(task))
            self.assertEqual(task["session_id"], task["result"]["session"]["session_id"])
            self.assertEqual("succeeded", task["result"]["session"]["status"])

    def test_z_executor_crash_unknown_recovery_no_replay(self):
        (self.directory / "action-entered").unlink(missing_ok=True)
        with self.client(0) as client:
            started = client.start_session(text="executor failure", device_id="device-local-01")
            session = started["session"]["session_id"]
        (self.directory / "block-next-action").touch()
        with ThreadPoolExecutor(1) as pool:
            def confirm():
                with self.client(1) as client:
                    return client.confirm_once(session)
            pending = pool.submit(confirm)
            deadline = time.monotonic() + 12
            while not (self.directory / "action-entered").exists():
                if pending.done():
                    pending.result()
                    self.fail("simulated action not entered")
                self.assertLess(time.monotonic(), deadline)
                time.sleep(.05)
            before = (self.directory / "simulated-actions.log").read_text()
            self.stop(self.executor)
            with self.assertRaises(LocalAgentApiError):
                pending.result(timeout=15)
        self.executor_socket.close()
        self.__class__.executor_socket = self.listener(self.executor_port)
        self.__class__.executor, new_identity = self.start_server("executor", self.executor_socket)
        self.wait_ready()
        with self.client(0) as client:
            current = client.get_session(session)["session"]
            self.assertEqual("recovery_required", current["status"])
            client.reconcile_session(session, external_effect="unknown", resume=False)
        self.assertEqual(before, (self.directory / "simulated-actions.log").read_text())
        with closing(sqlite3.connect(self.directory / "dispatch.sqlite3")) as db:
            self.assertGreaterEqual(db.execute("SELECT COUNT(*) FROM dispatch_tickets WHERE status='unknown'").fetchone()[0], 1)

    def test_zz_async_executor_crash_preserves_receipt_and_run_binding(self):
        (self.directory / "action-entered").unlink(missing_ok=True)
        (self.directory / "block-next-action").touch()
        with self.client(0) as first:
            accepted = first._request("POST", "/api/agent/generic-supervised/start-async",
                payload={"text":"async failure", "device_id":"device-local-01", "auto_advance":True}, read_only=False)
            task_id = accepted["task_id"]
        deadline = time.monotonic() + 12
        while not (self.directory / "action-entered").exists():
            self.assertLess(time.monotonic(), deadline)
            time.sleep(.05)
        before = (self.directory / "simulated-actions.log").read_text()
        self.stop(self.executor)
        self.executor_socket.close()
        self.__class__.executor_socket = self.listener(self.executor_port)
        self.__class__.executor, _ = self.start_server("executor", self.executor_socket)
        self.wait_ready()
        with self.client(1) as client:
            receipt = client._request("GET", "/api/agent/generic-supervised/start-async/{task_id}",
                                      path_params={"task_id":task_id}, read_only=True)
            self.assertEqual("async_start_outcome_unknown", receipt["error_detail"]["code"])
            self.assertEqual("recovery_required", client.get_session(receipt["session_id"])["session"]["status"])
            client.reconcile_session(receipt["session_id"], external_effect="unknown", resume=False)
        self.assertEqual(before, (self.directory / "simulated-actions.log").read_text())

    def test_l_pause_during_action_is_not_blocked_by_gateway(self):
        (self.directory / "action-entered").unlink(missing_ok=True)
        (self.directory / "block-next-action").touch()
        (self.directory / "nonterminal-action").touch()
        self.addCleanup((self.directory / "nonterminal-action").unlink, missing_ok=True)
        with self.client(0) as client:
            session = client.start_session(text="pause test", device_id="device-local-01")["session"]["session_id"]
        with ThreadPoolExecutor(1) as pool:
            def run():
                with self.client(0) as client:
                    return client.continue_automatic(session)
            pending = pool.submit(run)
            try:
                deadline = time.monotonic() + 12
                while not (self.directory / "action-entered").exists():
                    if pending.done():
                        pending.result()
                        self.fail("action not entered")
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.05)
                with self.client(1) as client:
                    client.pause_session(session)
            finally:
                (self.directory / "block-next-action").unlink(missing_ok=True)
                (self.directory / "nonterminal-action").unlink(missing_ok=True)
            pending.result(timeout=10)
        with self.client(1) as client:
            state = client.get_session(session)["session"]
            self.assertEqual("paused", state["status"])
            self.assertEqual(1, state["physical_actions"])
            client.cancel_session(session)

    def test_m_disabled_member_and_last_owner(self):
        with self.client(0) as owner, self.client(1, "worker-other") as other:
            other.control_plane_organization("local-org")
            owner.control_plane_update_principal("local-org", "worker-other", role="operator",
                status="disabled", expected_revision=0, idempotency_key="disable-other")
            with self.assertRaises(LocalAgentApiError) as denied:
                other.control_plane_organization("local-org")
            self.assertEqual(401, denied.exception.details.status_code)
            with self.assertRaises(LocalAgentApiError) as last_owner:
                owner.control_plane_update_principal("local-org", "local-owner", role="viewer",
                    status="active", expected_revision=0, idempotency_key="last-owner")
            self.assertEqual(409, last_owner.exception.details.status_code)

    def test_n_compare_and_swap_across_workers(self):
        with self.client(0) as client:
            client.control_plane_create_team("local-org", team_id="cas-team", name="CAS", member_ids=[],
                principal_token=client._principal_token, idempotency_key="cas-create")
        def update(index):
            with self.client(index) as client:
                try:
                    return client.control_plane_update_team("local-org", "cas-team", name=f"v{index}", expected_revision=0,
                        principal_token=client._principal_token, idempotency_key=f"cas-{index}")
                except LocalAgentApiError as error:
                    return error
        with ThreadPoolExecutor(2) as pool:
            replies = list(pool.map(update, (0,1)))
        self.assertEqual(1, sum(isinstance(r, dict) for r in replies))
        self.assertEqual(409, next(r for r in replies if isinstance(r, LocalAgentApiError)).details.status_code)
