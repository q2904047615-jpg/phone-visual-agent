"""Actual route functions in an isolated ASGI app, SQLite and fake upstream.

No socket, model call, camera or physical device is used. The extra dependency
is applied to a test router copy so the production application is unchanged.
"""
from types import SimpleNamespace
from unittest.mock import patch
from fastapi import Depends, FastAPI, Request
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
        self.repo._connection.execute("INSERT INTO control_plane_organizations VALUES('other-org','Other','other-owner','[]')")
        self.repo._connection.execute("INSERT INTO control_plane_principals VALUES('other-owner','other-org','owner','[]')")
        self.identities={}
        for user in ('alice','bob','other-owner'):
            value=self.repo.issue_identity_session(organization_id='other-org' if user=='other-owner' else 'local-org',user_id=user)
            self.identities[user]=value['principal_token']
        principal=self.repo.authenticate_identity_session(self.identities['alice'])
        self.repo.claim_resource('session','private-run',principal,device_id='alice-device')
        self.repo.claim_resource('device','alice-device',principal,device_id='alice-device')
        async def authorize(request: Request):
            await website_access.authorize(request,self.repo,web_app.verify_local_origin,'device-local-01')
        app=FastAPI()
        app.middleware('http')(web_app.disable_static_cache)
        app.include_router(web_app.app.router,dependencies=[Depends(authorize)])
        self.route_client=TestClient(app,base_url='http://127.0.0.1:8765')
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
        self.assertEqual(403,response.status_code,response.text)
        self.assertNotIn('principal_token',response.text)
