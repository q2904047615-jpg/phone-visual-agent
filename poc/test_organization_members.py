"""Member lifecycle through real routes and durable SQLite; no real devices."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import httpx
from starlette.requests import Request
from fastapi import HTTPException
import test_browser_identity as identity_fixture
from agent.interfaces import website_access
from agent.infrastructure.sqlite_control_plane_repository import SqliteControlPlaneRepository
from local_agent_api_client import LocalAgentApiClient
import web_app


class OrganizationMemberTests(unittest.TestCase):
    base = '/api/control-plane/organizations/local-org/principals'

    def setUp(self):
        identity_fixture.BrowserIdentityTests.setUp(self)
        self.repo.initialize_website_ownership('device-local-01')
        self.tokens = {}
        for user, role in [('alice','operator'), ('admin','admin'), ('viewer','viewer')]:
            self.repo._connection.execute('INSERT INTO control_plane_principals VALUES(?,?,?,?)', (user,'local-org',role,'[]'))
        for user in ('local-owner','alice','admin','viewer'):
            self.tokens[user] = self.repo.issue_identity_session(organization_id='local-org',user_id=user)['principal_token']
        for p in [patch.object(web_app, '_active_session_status', return_value=[]),
                  patch.object(web_app, '_device_assignment_unavailable', return_value='')]:
            p.start(); self.addCleanup(p.stop)

    def auth(self, actor='local-owner', key='write'):
        return {**self.headers,'X-Control-Plane-Token':self.tokens[actor],'Idempotency-Key':key}

    def update(self, user='alice', role='operator', status='active', revision=0, actor='local-owner', key='update'):
        return self.client.put(self.base+'/'+user, headers=self.auth(actor,key),
            json={'role':role,'status':status,'expected_revision':revision})

    def create(self, user, role, actor='local-owner'):
        response = self.client.post(self.base, headers=self.auth(actor,'create-'+user),json={'user_id':user,'role':role})
        if response.status_code == 201 and role == 'owner':
            self.repo.update_github_identity(organization_id='local-org',principal_user_id='local-owner',
                subject='43',user_id=user,expected_revision=0,idempotency_key='fixture-owner-link')
        return response

    def test_owner_admin_and_member_hierarchy_on_old_and_new_routes(self):
        for role in ['owner','admin']:
            self.assertEqual(403,self.create('forbidden-'+role,role,'admin').status_code)
        self.assertEqual(201,self.create('new-viewer','viewer','admin').status_code)
        self.assertEqual(200,self.update(role='viewer',actor='admin').status_code)
        self.assertEqual(403,self.update(user='admin',role='viewer',actor='admin',key='self').status_code)
        self.assertEqual(403,self.update(user='local-owner',role='viewer',actor='admin',key='owner').status_code)
        self.assertEqual(403,self.update(role='owner',actor='admin',key='promote').status_code)
        for actor in ['viewer','alice']:
            self.assertEqual(403,self.client.get(self.base,headers=self.auth(actor)).status_code)
            self.assertEqual(403,self.create('forbidden-'+actor,'viewer',actor).status_code)
            self.assertEqual(403,self.update(actor=actor).status_code)
        self.assertEqual(403,self.client.get(self.base.replace('local-org','foreign'),headers=self.auth()).status_code)
        self.assertEqual(404,self.update(user='foreign-user').status_code)
        listing=self.client.get(self.base,headers=self.auth('admin')).json()['data']
        self.assertEqual(['viewer','operator'],listing['create_roles'])
        self.assertEqual([],next(p for p in listing['principals'] if p['user_id']=='local-owner')['editable_roles'])

    def test_last_owner_and_designated_owner_transfer_are_atomic(self):
        for role,status in [('viewer','active'),('owner','disabled')]:
            r=self.update(user='local-owner',role=role,status=status,key=status)
            self.assertEqual(409,r.status_code,r.text)
            self.assertEqual('last_owner_required',r.json()['detail']['code'])
        self.assertEqual(201,self.create('second-owner','owner').status_code)
        self.assertEqual(200,self.update(user='local-owner',role='viewer',key='transfer').status_code)
        self.assertEqual('second-owner',self.repo.get_organization('local-org')['owner_user_id'])
        self.assertEqual(403,self.client.get(self.base,headers=self.auth()).status_code)

    def test_disable_revokes_all_sessions_restore_never_revives_and_preserves_ownership(self):
        second=self.repo.issue_identity_session(organization_id='local-org',user_id='alice')['principal_token']
        principal=self.repo.authenticate_identity_session(second)
        self.repo.claim_resource('session','historical',principal,device_id='device-local-01')
        self.assertEqual(200,self.update(status='disabled').status_code)
        for token in [self.tokens['alice'],second]: self.assertIsNone(self.repo.authenticate_identity_session(token))
        self.assertIsNotNone(self.repo.owned_resource('session','historical',principal))
        with self.assertRaises(KeyError): self.repo.issue_identity_session(organization_id='local-org',user_id='alice')
        r=self.client.put('/api/control-plane/organizations/local-org/devices/device-local-01/assignment',
            headers=self.auth(key='disabled-assignee'),json={'user_id':'alice','expected_revision':0})
        self.assertEqual(422,r.status_code,r.text)
        self.assertEqual(200,self.update(status='active',revision=1,key='restore').status_code)
        self.assertIsNone(self.repo.authenticate_identity_session(second))
        self.assertTrue(self.repo.issue_identity_session(organization_id='local-org',user_id='alice')['principal_token'])
        self.assertEqual('local-owner',self.repo.device_assignments('local-org')[0]['user_id'])

    def test_role_change_applies_to_existing_token_and_dispatch_recheck(self):
        req=Request({'type':'http','method':'POST','path':'/api/agent/chat','query_string':b'',
            'headers':[(k.lower().encode(),v.encode()) for k,v in self.auth('alice').items()],
            'route':SimpleNamespace(path='/api/agent/chat')})
        req.state.principal=self.repo.authenticate_identity_session(self.tokens['alice'])
        self.assertEqual(200,self.update(role='viewer').status_code)
        self.assertNotIn('task.create',self.repo.authenticate_identity_session(self.tokens['alice'])['permissions'])
        with self.assertRaises(HTTPException) as caught:
            with website_access.member_access(self.repo,req): self.fail('stale request dispatched')
        self.assertEqual(403,caught.exception.status_code)
        self.assertEqual(200,self.client.get('/api/control-plane/organizations/local-org',headers=self.auth('alice')).status_code)
        self.assertEqual(403,self.client.post('/api/agent/chat',headers=self.auth('alice'),json={'text':'hello'}).status_code)

    def test_revision_idempotency_durability_and_audit(self):
        self.assertEqual(200,self.update(role='viewer').status_code)
        replay=self.update(role='viewer')
        self.assertTrue(replay.json()['idempotent_replay'])
        self.assertEqual(409,self.update(role='admin').status_code)
        self.assertEqual(409,self.update(role='admin',key='stale').status_code)
        reopened=SqliteControlPlaneRepository(self.repo.database_path)
        try:
            alice=next(m for m in reopened.list_principals('local-org') if m['user_id']=='alice')
            self.assertEqual(('viewer',1),(alice['role'],alice['revision']))
            events=[e for e in reopened.read_audit('local-org',limit=100)['entries'] if e['resource']=='principal.update' and e['outcome']=='updated']
            self.assertEqual(1,len(events)); self.assertEqual('local-owner',events[0]['actor'])
            self.assertEqual('operator',events[0]['metadata']['old_role'])
        finally: reopened.close()

    def test_busy_member_cannot_be_disabled_and_retry_does_not_lose_history(self):
        principal=self.repo.authenticate_identity_session(self.tokens['alice'])
        self.repo.claim_resource('session','paused-run',principal,device_id='device-local-01')
        with patch.object(web_app,'_active_session_status',return_value=[{'session_id':'paused-run','device_id':'device-local-01'}]):
            r=self.update(status='disabled')
            self.assertEqual(409,r.status_code,r.text)
            self.assertEqual('member_has_active_work',r.json()['detail']['code'])
        self.assertIsNotNone(self.repo.authenticate_identity_session(self.tokens['alice']))
        req=Request({'type':'http','method':'POST','path':'/api/agent/chat','query_string':b'',
            'headers':[(k.lower().encode(),v.encode()) for k,v in self.auth('alice').items()],
            'route':SimpleNamespace(path='/api/agent/chat')})
        req.state.principal=principal
        with website_access.member_access(self.repo,req):
            self.assertEqual(409,self.update(status='disabled',key='inflight').status_code)
        self.assertEqual(200,self.update(status='disabled').status_code)
        self.assertEqual(0,website_access.inflight_member_writes[('local-org','alice')])

    def test_concurrent_updates_cannot_overwrite_each_other(self):
        with ThreadPoolExecutor() as pool:
            replies=list(pool.map(lambda role:self.update(role=role,key=role),['viewer','admin']))
        self.assertEqual([200,409],sorted(r.status_code for r in replies))

    def test_separate_connections_cannot_remove_both_owners(self):
        self.assertEqual(201,self.create('second-owner','owner').status_code)
        other=SqliteControlPlaneRepository(self.repo.database_path)
        try:
            def demote(pair):
                store,user=pair
                try:
                    return store.update_principal(organization_id='local-org',principal_user_id=user,user_id=user,
                        role='viewer',status='active',expected_revision=0,idempotency_key=user)[0]
                except ValueError as error:
                    return str(error)
            with ThreadPoolExecutor() as pool:
                results=list(pool.map(demote,[(self.repo,'local-owner'),(other,'second-owner')]))
            self.assertEqual(1,results.count(200))
            self.assertEqual(1,results.count('last_owner_required'))
        finally: other.close()

    def test_disabled_mapped_github_identity_cannot_sign_in(self):
        # The existing simulated provider maps subject 42 to local-owner.
        self.assertEqual(201,self.create('second-owner','owner').status_code)
        self.assertEqual(200,self.update(user='local-owner',role='owner',status='disabled').status_code)
        r=self.client.post('/api/control-plane/auth/external-sessions',headers=self.headers,
            json={'provider_id':'github','assertion':{'access_token':'test-github-token'}})
        self.assertEqual(403,r.status_code,r.text)
        self.assertEqual('principal_disabled',r.json()['detail']['code'])
        state=identity_fixture.BrowserIdentityTests.start(self)
        callback=identity_fixture.BrowserIdentityTests.complete(self,state)
        self.assertEqual('/?login_error=principal_disabled',callback.headers['location'])
        self.assertIsNone(self.client.cookies.get(web_app._BROWSER_SESSION_COOKIE))

    def test_required_fields_and_official_sdk(self):
        for body in [{'role':'viewer','status':'active'}, {'role':'viewer','status':'active','expected_revision':True}]:
            self.assertEqual(422,self.client.put(self.base+'/alice',headers=self.auth(),json=body).status_code)
        def transport(request):
            r=self.client.request(request.method,request.url.raw_path.decode(),headers=dict(request.headers),content=request.content)
            return httpx.Response(r.status_code,content=r.content,headers=dict(r.headers))
        with LocalAgentApiClient(transport=httpx.MockTransport(transport),principal_token=self.tokens['local-owner']) as sdk:
            result=sdk.control_plane_update_principal('local-org','alice',role='viewer',status='active',expected_revision=0,idempotency_key='sdk')
            self.assertEqual('viewer',result['data']['principal']['role'])
