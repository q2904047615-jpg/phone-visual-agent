"""Assignment ownership, admission races and durability through production HTTP."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch
import httpx
import test_browser_identity as identity_fixture
from agent.infrastructure.sqlite_control_plane_repository import SqliteControlPlaneRepository
from local_agent_api_client import LocalAgentApiClient
from agent.interfaces import website_access
import web_app


class DeviceAssignmentTests(identity_fixture.BrowserIdentityTests):
    base = '/api/control-plane/organizations/local-org/devices'
    device = 'device-local-01'

    def setUp(self):
        super().setUp()
        self.repo.initialize_website_ownership(self.device)
        self.tokens = {}
        for user, role in [('alice','operator'), ('admin','admin'), ('viewer','viewer')]:
            self.repo._connection.execute('INSERT INTO control_plane_principals VALUES(?,?,?,?)',(user,'local-org',role,'[]'))
        for user in ('local-owner','alice','admin','viewer'):
            self.tokens[user] = self.repo.issue_identity_session(organization_id='local-org',user_id=user)['principal_token']
        for p in [patch.object(web_app,'_active_session_status',return_value=[]),
                  patch.object(web_app.runtime.device_task_registry,'active_session',return_value=None),
                  patch.object(web_app._GENERIC_START_TASKS,'active_for',return_value=None),
                  patch.object(web_app._LUCKY_BAG_MONITOR,'list',return_value=[]),
                  patch.object(web_app._QISHUI_AD_TEST_MONITOR,'list',return_value=[]),
                  patch.object(web_app._MANUAL_MOTION_MONITOR,'list',return_value=[]),
                  patch.object(web_app.runtime.capability_acceptance_manager,'snapshots',return_value=[]),
                  patch.object(web_app.runtime.device_controllers,'descriptors',return_value=[{'device_id':self.device}]),
                  patch.object(web_app.runtime,'controller_for_device',return_value=SimpleNamespace(device_status=lambda:{'busy':False}))]:
            p.start(); self.addCleanup(p.stop)

    def auth(self, user='local-owner'):
        return {**self.headers,'X-Control-Plane-Token':self.tokens[user]}

    def assign(self, user, revision, key, actor='local-owner', device=None):
        return self.client.put(self.base + f'/{device or self.device}/assignment',
            headers={**self.auth(actor),'Idempotency-Key':key}, json={'user_id':user,'expected_revision':revision})

    def test_assign_unassign_reassign_preserves_history_and_revokes_preview(self):
        owner=self.repo.authenticate_identity_session(self.tokens['local-owner'])
        self.repo.claim_resource('session','old-owner-run',owner,device_id=self.device)
        first=self.assign('alice',0,'first',actor='admin')
        self.assertEqual(200,first.status_code,first.text)
        with patch.object(web_app.runtime,'capture_preview',return_value=(b'jpg',False)) as capture:
            self.assertEqual(403,self.client.get('/api/preview.jpg',params={'device_id':self.device},headers=self.auth()).status_code)
            capture.assert_not_called()
            self.assertEqual(200,self.client.get('/api/preview.jpg',params={'device_id':self.device},headers=self.auth('alice')).status_code)
        self.assertEqual(404,self.client.get('/api/agent/generic-supervised/old-owner-run',headers=self.auth('alice')).status_code)
        self.assertIsNotNone(self.repo.owned_resource('session','old-owner-run',owner))
        self.assertEqual(200,self.assign(None,1,'unassign').status_code)
        self.assertEqual(403,self.client.get('/api/preview.jpg',params={'device_id':self.device},headers=self.auth('alice')).status_code)
        self.assertEqual(200,self.assign('local-owner',2,'restore').status_code)
        with self.repo._lock:
            rows=self.repo._connection.execute("select actor,outcome from control_plane_audit where resource='device.assign'").fetchall()
        self.assertIn(('admin','assigned'),[tuple(r) for r in rows])
        reopened=SqliteControlPlaneRepository(self.repo.database_path)
        try: self.assertEqual(3,reopened.device_assignments('local-org')[0]['revision'])
        finally: reopened.close()

    def test_inventory_and_writes_enforce_role_org_member_and_registry(self):
        for actor in ('alice','viewer'):
            self.assertEqual(403,self.client.get(self.base,headers=self.auth(actor)).status_code)
            self.assertEqual(403,self.assign(actor,0,'denied-'+actor,actor=actor).status_code)
        listing=self.client.get(self.base,headers=self.auth('admin'))
        self.assertEqual(200,listing.status_code)
        self.assertEqual({'device_id','organization_id','user_id','revision','registered'},set(listing.json()['data']['devices'][0]))
        self.assertEqual(403,self.client.get(self.base.replace('local-org','foreign'),headers=self.auth('admin')).status_code)
        self.assertEqual(422,self.assign('unknown',0,'unknown').status_code)
        self.assertEqual(404,self.assign('alice',0,'missing',device='disabled-phone').status_code)
        self.assertEqual(422,self.client.put(self.base+f'/{self.device}/assignment',headers=self.auth(),json={'user_id':'alice','expected_revision':True}).status_code)

    def test_idempotency_conflict_and_compare_and_swap(self):
        self.assertEqual(200,self.assign('alice',0,'repeat').status_code)
        with patch.object(web_app,'_device_assignment_unavailable',return_value='device_has_active_work'):
            response=self.assign('alice',0,'repeat')
        self.assertTrue(response.json()['idempotent_replay'])
        self.assertEqual(409,self.assign(None,0,'repeat').status_code)
        self.assertEqual(409,self.assign(None,0,'stale').status_code)
        self.assertEqual(1,self.repo.device_assignments('local-org')[0]['revision'])

    def test_omitted_assignee_or_idempotency_key_cannot_unassign(self):
        path=self.base+f'/{self.device}/assignment'
        missing_user=self.client.put(path,headers={**self.auth(),'Idempotency-Key':'missing-user'},json={'expected_revision':0})
        self.assertEqual(422,missing_user.status_code)
        missing_key=self.client.put(path,headers=self.auth(),json={'user_id':None,'expected_revision':0})
        self.assertEqual(422,missing_key.status_code)
        self.assertEqual('local-owner',self.repo.device_assignments('local-org')[0]['user_id'])
        self.assertEqual(0,self.repo.device_assignments('local-org')[0]['revision'])

    def test_active_and_paused_work_cannot_be_reassigned(self):
        for manager, method, value in [
            (web_app,'_active_session_status',[{'device_id':self.device,'status':'recovery_required'}]),
            (web_app._GENERIC_START_TASKS,'active_for',{'status':'queued'}),
            (web_app._LUCKY_BAG_MONITOR,'list',[SimpleNamespace(profile=SimpleNamespace(device_id=self.device),status='paused')]),
            (web_app.runtime.capability_acceptance_manager,'snapshots',[{'device_id':self.device,'session':{'status':'awaiting_confirmation'}}])]:
            with self.subTest(method=method), patch.object(manager,method,return_value=value):
                result=self.assign('alice',0,'busy-'+method)
                self.assertEqual(409,result.status_code,result.text)
        self.assertEqual('local-owner',self.repo.device_assignments('local-org')[0]['user_id'])

    def test_inflight_read_finishes_before_transfer_and_old_request_is_rechecked(self):
        entered,release=Event(),Event()
        def capture(*args,**kwargs):
            entered.set(); self.assertTrue(release.wait(5)); return b'jpg',False
        with patch.object(web_app.runtime,'capture_preview',side_effect=capture), ThreadPoolExecutor() as pool:
            future=pool.submit(self.client.get,'/api/preview.jpg',params={'device_id':self.device},headers=self.auth())
            try:
                self.assertTrue(entered.wait(5))
                self.assertEqual(409,self.assign('alice',0,'inflight').status_code)
            finally: release.set()
            self.assertEqual(200,future.result().status_code)
        self.assertEqual(200,self.assign('alice',0,'after').status_code)
        self.assertEqual(0,website_access.inflight_devices[self.device])

    def test_sdk_uses_production_openapi_and_principal(self):
        def transport(request):
            response=self.client.request(request.method,request.url.raw_path.decode(),headers=dict(request.headers),content=request.content)
            return httpx.Response(response.status_code,content=response.content,headers=dict(response.headers))
        with LocalAgentApiClient(transport=httpx.MockTransport(transport),principal_token=self.tokens['local-owner']) as sdk:
            self.assertEqual(1,len(sdk.control_plane_devices('local-org')['data']['devices']))
            response=sdk.control_plane_assign_device('local-org',self.device,user_id='alice',expected_revision=0,idempotency_key='sdk')
            self.assertEqual('alice',response['data']['assignment']['user_id'])

    def test_authorized_request_is_rechecked_after_transfer_and_stream_stops(self):
        import asyncio
        from starlette.requests import Request
        request=Request({'type':'http','method':'GET','path':'/api/preview.mjpg',
            'query_string':b'device_id=device-local-01','headers':[(k.lower().encode(),v.encode()) for k,v in self.auth().items()],
            'route':SimpleNamespace(path='/api/preview.mjpg')})
        with patch.object(web_app.runtime,'capture_preview',return_value=(b'frame',False)) as capture, patch.object(web_app.time,'sleep'):
            response=web_app.preview_mjpg(self.device,request)
            async def run():
                first=await anext(response.body_iterator)
                self.assertIn(b'frame',first)
                self.assertEqual(200,self.assign('alice',0,'stream-transfer').status_code)
                with self.assertRaises(StopAsyncIteration): await anext(response.body_iterator)
            asyncio.run(run())
            self.assertEqual(1,capture.call_count)
            with self.assertRaises(web_app.HTTPException) as failure:
                with website_access.device_access(self.repo,request,{self.device}):
                    self.fail('Old ownership reached dispatch')
            self.assertEqual(403,failure.exception.status_code)
