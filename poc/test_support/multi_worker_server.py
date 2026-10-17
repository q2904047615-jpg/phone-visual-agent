"""Spawn-only isolated test servers. No production databases or remote models."""
import os
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace


def try_second_executor(directory, reply):
    os.environ["PHONE_AGENT_STATE_DIR"] = directory
    from single_executor import create_app
    try:
        create_app()
    except RuntimeError:
        reply.send({"rejected":True, "runtime_imported":"web_app" in __import__('sys').modules})
    else:
        reply.send({"rejected":False})


def run_server(role, directory, listener, executor_port, ready):
    directory = Path(directory)
    os.environ.update(ROBOT_WEB_MOCK="1", ROBOT_WEB_NO_BROWSER="1",
                      PHONE_AGENT_STATE_DIR=str(directory),
                      PHONE_VISUAL_AGENT_LEASE_DIR=str(directory / "leases"),
                      PHONE_AGENT_EXECUTOR_URL=f"http://127.0.0.1:{executor_port}")
    for key in list(os.environ):
        if key.startswith(("DASHSCOPE_", "GITHUB_", "GMAIL_")) or key == "PUBLIC_BASE_URL":
            os.environ.pop(key, None)
    import uvicorn
    if role == "api":
        from multi_worker_api import create_app
        app = create_app()
        ready.send({"pid": os.getpid(), "role": "api", "runtime_imported": "web_app" in __import__('sys').modules})
    else:
        from single_executor import create_app
        app = create_app()
        import web_app
        from agent.infrastructure.github_oauth_provider import GitHubOAuthProvider
        from agent.infrastructure.control_plane_identity import ExternalIdentityProviderRegistry
        from agent.application.qwen_visual_decision import QwenVisualDecisionObserver
        from test_single_visual_loop import Adapter, Observation, scene, decision
        import httpx
        repo = web_app.runtime.control_plane_repository
        repo.initialize_website_ownership("device-local-01")
        repo.import_github_mappings({"42": {"organization_id":"local-org", "user_id":"local-owner"}})
        for user, role_name in (("worker-admin", "admin"), ("worker-other", "operator")):
            if user not in {p["user_id"] for p in repo.list_principals("local-org")}:
                repo.create_principal(organization_id="local-org", principal_user_id="local-owner",
                                      user_id=user, role=role_name, team_ids=(),
                                      idempotency_key=user, request_hash=user)
        identities = {user: repo.issue_identity_session(organization_id="local-org", user_id=user)
                      for user in (p["user_id"] for p in repo.list_principals("local-org")
                                   if p["status"] == "active")}
        principal = repo.authenticate_identity_session(identities["local-owner"]["principal_token"])
        repo.claim_resource("device", "mock-device-2", principal, device_id="mock-device-2")
        def github(request):
            if request.url.host == "github.com":
                return httpx.Response(200, json={"access_token":"isolated-upstream-token"})
            return httpx.Response(200, json={"app":{"client_id":"fixture-client"}, "user":{"id":42}})
        provider = GitHubOAuthProvider(client_id="fixture-client", client_secret="fixture-secret",
                                       mapping_resolver=repo.github_identity, transport=httpx.MockTransport(github))
        web_app.control_plane_identity_registry = ExternalIdentityProviderRegistry()
        web_app.control_plane_identity_registry.register(provider)

        class SimulatedAdapter(Adapter):
            def __init__(self, device_id):
                rows = [(scene(0), decision("home")), (scene(1), decision())]
                if (directory / "nonterminal-action").exists():
                    rows = [(scene(0), decision("home")), (scene(1), decision("home")),
                            (scene(2), decision())]
                super().__init__(rows)
                self.device_id = device_id

            def text_transport_profile(self):
                return replace(super().text_transport_profile(), device_id=self.device_id)

            def execute(self, **kwargs):
                with (directory / "simulated-actions.log").open("a") as output:
                    output.write("simulated action\n")
                if (directory / "block-next-action").exists():
                    (directory / "action-entered").touch()
                    deadline = time.monotonic() + 30
                    while (directory / "block-next-action").exists() and time.monotonic() < deadline:
                        time.sleep(.05)
                return super().execute(**kwargs)

        orchestrator = web_app.runtime.universal_agent_orchestrator
        orchestrator.adapter_factory = SimulatedAdapter
        # Check the actual constructor attribute names rather than replacing the service.
        orchestrator.model_gateway = QwenVisualDecisionObserver(SimpleNamespace(status=lambda:{"model":"offline"}),
                                                                trusted_observation_frame_validator=lambda *a, **k: None)
        orchestrator.trusted_observation_factory = Observation
        service = web_app.runtime.universal_agent_session_service
        service._ensure_device_ready = lambda _: None
        service._exclusive_device_session = lambda _: nullcontext()
        service._begin_new_task = lambda _: None
        ready.send({"pid":os.getpid(), "role":"executor", "identities":identities})
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False,
                                          proxy_headers=True, forwarded_allow_ips="127.0.0.1"))
    server.run(sockets=[listener])
