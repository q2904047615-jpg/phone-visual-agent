"""Production HTTP routes with isolated SQLite and fake upstream; no devices."""
from types import SimpleNamespace
from unittest.mock import patch
from starlette.requests import Request
import asyncio
import httpx
from local_agent_api_client import LocalAgentApiClient
from fastapi.testclient import TestClient
import test_browser_identity as identity_fixture
from agent.interfaces import website_access
import web_app


class WebsiteRouteIsolationTests(identity_fixture.BrowserIdentityTests):
    def setUp(self):
        super().setUp()
        self.repo.initialize_website_ownership('device-local-01')
        self.repo._connection.execute("INSERT INTO control_plane_principals VALUES('alice','local-org','operator','[]')")
        self.repo._connection.execute("INSERT INTO control_plane_principals VALUES('bob','local-org','admin','[]')")
        self.repo._connection.execute("INSERT INTO control_plane_principals VALUES('viewer','local-org','viewer','[]')")
        self.repo._connection.execute("INSERT INTO control_plane_organizations VALUES('other-org','Other','other-owner','[]')")
        self.repo._connection.execute("INSERT INTO control_plane_principals VALUES('other-owner','other-org','owner','[]')")
        self.identities={}
        for user in ('alice','bob','other-owner','viewer'):
            value=self.repo.issue_identity_session(organization_id='other-org' if user=='other-owner' else 'local-org',user_id=user)
            self.identities[user]=value['principal_token']
        principal=self.repo.authenticate_identity_session(self.identities['alice'])
        self.repo.claim_resource('session','private-run',principal,device_id='alice-device')
        self.repo.claim_resource('device','alice-device',principal,device_id='alice-device')
        self.route_client=TestClient(web_app.app,base_url='http://127.0.0.1:8765')
        self.addCleanup(self.route_client.close)
        self.client=self.route_client

    def auth(self,user):
        return {**self.headers,'X-Control-Plane-Token':self.identities[user]}

    def test_own_session_read_and_cross_user_reads_never_enter_runtime(self):
        session=SimpleNamespace(snapshot=lambda: {'session_id':'private-run','goal':'alice-private-goal'},run_dir=__import__('pathlib').Path('private'))
        with patch.object(web_app,'_require_generic_supervised_session',return_value=session) as require:
            route='/api/agent/generic-supervised/private-run'
            own=self.route_client.get(route,headers=self.auth('alice'))
            self.assertEqual(200,own.status_code,own.text)
            self.assertIn('alice-private-goal',own.text)
            require.reset_mock()
            for user in ('bob','other-owner'):
                denied=self.route_client.get(route,headers=self.auth(user))
                self.assertEqual(404,denied.status_code,denied.text)
                self.assertNotIn('alice-private-goal',denied.text)
            require.assert_not_called()

    def test_anonymous_preview_and_cross_device_write_never_touch_hardware(self):
        with patch.object(web_app.runtime,'capture_preview') as capture, patch.object(web_app.runtime,'controller_for_device') as controller:
            for user in (None,'bob','other-owner'):
                headers=self.auth(user) if user else self.headers
                response=self.route_client.get('/api/preview.jpg?device_id=alice-device',headers=headers)
                self.assertEqual(403 if user else 401,response.status_code,response.text)
            denied=self.route_client.post('/api/device/alice-device/directional-swipe',headers=self.auth('bob'),json={'direction':'left'})
            self.assertEqual(403,denied.status_code,denied.text)
            capture.assert_not_called(); controller.assert_not_called()

    def test_replay_and_evidence_cross_user_rejected_before_file_access(self):
        principal=self.repo.authenticate_identity_session(self.identities['alice'])
        self.repo.claim_resource('trial','private-trial',principal,device_id='alice-device')
        for route in ('/api/agent/generic-supervised/private-run/events',
                      '/api/capability-acceptance/private-trial/evidence/before/0'):
            response=self.route_client.get(route,headers=self.auth('bob'))
            self.assertEqual(404,response.status_code,response.text)

    def test_cross_org_read_denied_before_projection(self):
        response=self.route_client.get('/api/control-plane/organizations/other-org/principals',headers=self.auth('bob'))
        self.assertEqual(403,response.status_code,response.text)

    def test_authenticated_operator_cannot_issue_another_users_session(self):
        response=self.route_client.post('/api/control-plane/auth/sessions',headers=self.auth('alice'),
                                        json={'organization_id':'local-org','user_id':'local-owner'})
        self.assertIn(response.status_code,(404,405),response.text)
        self.assertNotIn('principal_token',response.text)
        self.assertNotIn('/api/control-plane/auth/sessions',self.route_client.get('/openapi.json').json()['paths'])

    def test_github_cookie_protects_business_routes_and_logout_revokes_them(self):
        self.complete(self.start())
        cookie = self.client.cookies.get(web_app._BROWSER_SESSION_COOKIE)
        self.assertTrue(any(c.path == '/api' and c.name == web_app._BROWSER_SESSION_COOKIE for c in self.client.cookies.jar))
        principal=self.repo.authenticate_identity_session(cookie)
        self.repo.claim_resource('session','cookie-owned',principal,device_id='device-local-01')
        session=SimpleNamespace(snapshot=lambda: {'session_id':'cookie-owned'},run_dir=__import__('pathlib').Path('private'))
        with patch.object(web_app,'_require_generic_supervised_session',return_value=session):
            self.assertEqual(200,self.client.get('/api/agent/generic-supervised/cookie-owned',headers=self.headers).status_code)
            self.assertEqual(401,self.client.get('/api/agent/generic-supervised/cookie-owned',headers={**self.headers,'X-Control-Plane-Token':'invalid'}).status_code)
            self.assertEqual(200,self.client.post('/api/control-plane/auth/browser-logout',headers=self.headers).status_code)
            self.assertEqual(401,self.client.get('/api/agent/generic-supervised/cookie-owned',headers=self.headers).status_code)

    def test_no_devices_returns_empty_status_and_stop_never_touches_others(self):
        with patch.object(web_app.runtime,'controller_for_device') as controller, patch.object(web_app.runtime.controller,'device_status') as status:
            response=self.client.get('/api/device',headers=self.auth('bob'))
            self.assertEqual(200,response.status_code,response.text)
            self.assertEqual([],response.json()['devices'])
            self.assertEqual([],response.json()['active_tasks'])
            stopped=self.client.post('/api/stop',headers=self.auth('bob'))
            self.assertEqual([],stopped.json()['stopped_device_ids'])
            controller.assert_not_called(); status.assert_not_called()

    def test_lists_and_audit_filter_before_returning(self):
        alice=self.repo.authenticate_identity_session(self.identities['alice'])
        bob=self.repo.authenticate_identity_session(self.identities['bob'])
        self.repo.claim_resource('trial','alice-trial',alice,device_id='alice-device')
        self.repo.claim_resource('trial','bob-trial',bob,device_id='bob-device')
        with patch.object(web_app.runtime.capability_acceptance_manager,'snapshots',return_value=[
            {'trial_id':'alice-trial','goal':'alice-only'},{'trial_id':'bob-trial','goal':'bob-private'}]):
            response=self.client.get('/api/capability-acceptance',headers=self.auth('alice'))
            self.assertEqual([{'trial_id':'alice-trial','goal':'alice-only'}],response.json()['trials'])
        self.repo.record_audit(organization_id='local-org',resource='fixture',outcome='ok',request_id='other',actor='bob',metadata={'value':'bob-private'})
        self.repo.record_audit(organization_id='local-org',resource='fixture',outcome='ok',request_id='own',actor='alice')
        page=self.client.get('/api/control-plane/organizations/local-org/audit?limit=1',headers=self.auth('alice'))
        self.assertEqual(200,page.status_code,page.text)
        self.assertEqual('alice',page.json()['data']['entries'][0]['actor'])
        self.assertNotIn('bob-private',page.text)

    def test_viewer_cannot_write_or_restart_but_can_read_own_data(self):
        viewer=self.repo.authenticate_identity_session(self.identities['viewer'])
        self.repo.claim_resource('session','viewer-task',viewer)
        with patch.object(web_app,'_require_generic_supervised_session',return_value=SimpleNamespace(snapshot=lambda:{'session_id':'viewer-task'},run_dir=__import__('pathlib').Path('private'))):
            self.assertEqual(200,self.client.get('/api/agent/generic-supervised/viewer-task',headers=self.auth('viewer')).status_code)
        for path in ('/api/agent/generic-supervised/start','/api/session/reload'):
            response=self.client.post(path,headers=self.auth('viewer'),json={'text':'no action','device_id':'alice-device'})
            self.assertEqual(403,response.status_code,response.text)

    def test_main_start_claims_owner_before_runtime_dispatch(self):
        principal=self.repo.authenticate_identity_session(self.identities['alice'])
        def dispatch(command):
            self.assertIsNotNone(self.repo.owned_resource('session',command.session_id,principal))
            session=SimpleNamespace(snapshot=lambda:{'session_id':command.session_id,'device_id':command.device_id})
            return SimpleNamespace(session=session,automatic_progress={'physical_actions':0})
        with patch.object(web_app.runtime.universal_agent_session_service,'start',side_effect=dispatch) as start, patch.object(web_app,'_write_generic_supervised_report',return_value='test-report'):
            response=self.client.post('/api/agent/generic-supervised/start',headers=self.auth('alice'),json={'text':'fixture','device_id':'alice-device','auto_advance':False})
            self.assertEqual(200,response.status_code,response.text)
            start.assert_called_once()
        denied=self.client.get('/api/agent/generic-supervised/'+response.json()['session']['session_id'],headers=self.auth('bob'))
        self.assertEqual(404,denied.status_code)

    def test_async_ticket_is_private_even_before_session_exists(self):
        with patch.object(web_app._GENERIC_START_TASKS,'submit'):
            response=self.client.post('/api/agent/generic-supervised/start-async',headers=self.auth('alice'),json={'text':'fixture','device_id':'alice-device','auto_advance':False})
            self.assertEqual(200,response.status_code,response.text)
        route='/api/agent/generic-supervised/start-async/'+response.json()['task_id']
        self.assertEqual(200,self.client.get(route,headers=self.auth('alice')).status_code)
        self.assertEqual(404,self.client.get(route,headers=self.auth('bob')).status_code)

    def test_stream_rechecks_revocation_without_capturing_another_frame(self):
        token=self.identities['alice']
        request=Request({'type':'http','method':'GET','headers':[(b'x-control-plane-token',token.encode())]})
        with patch.object(web_app.runtime,'controller_for_device'),patch.object(web_app.runtime,'capture_preview',return_value=(b'fixture-frame',False)) as capture,patch.object(web_app.time,'sleep'):
            response=web_app.preview_mjpg('alice-device',request)
            self.assertIn(b'fixture-frame',asyncio.run(response.body_iterator.__anext__()))
            self.repo.revoke_identity_session(self.repo.authenticate_identity_session(token)['session_id'])
            with self.assertRaises(StopAsyncIteration): asyncio.run(response.body_iterator.__anext__())
            self.assertEqual(1,capture.call_count)

    def test_official_client_sends_identity_and_works_on_production_routes(self):
        def transport(request):
            reply=self.client.request(request.method,str(request.url),headers=dict(request.headers),content=request.content)
            return httpx.Response(reply.status_code,json=reply.json())
        with LocalAgentApiClient(principal_token=self.identities['alice'],transport=httpx.MockTransport(transport)) as client:
            self.assertTrue(client.bootstrap()['openapi_verified'])
            self.assertEqual('local-org',client.control_plane_organization('local-org')['data']['organization']['organization_id'])

    def test_same_host_other_port_cannot_mutate_cookie_session(self):
        self.complete(self.start())
        response=self.client.post('/api/control-plane/auth/browser-logout',headers={**self.headers,'Origin':'http://127.0.0.1:9999'})
        self.assertEqual(403,response.status_code)
        self.assertTrue(self.status().json()['data']['authenticated'])

    def test_shared_model_diagnostics_do_not_leak_last_user_data(self):
        self.complete(self.start())
        with patch.object(web_app.runtime.vision_provider,'status',return_value={
            'configured':True,'model':'test-model','last_usage':{'private':'other-user'},
            'last_request_id':'other-request','usage_totals':{'tokens':1234}}), patch.object(
                web_app.runtime.generic_scene_observer,'status',return_value={'configured':True,'last_scene_enum_values':{'value':'other-user'}}):
            response=self.client.get('/api/device',headers=self.headers)
            self.assertEqual(200,response.status_code,response.text)
            self.assertNotIn('other-user',response.text)
            self.assertNotIn('other-request',response.text)
            self.assertNotIn('usage_totals',response.text)
