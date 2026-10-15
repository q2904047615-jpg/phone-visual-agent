"""Real HTTP/repository integration, with GitHub and identities isolated."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import httpx
import test_browser_identity as fixture
from agent.infrastructure.github_oauth_provider import GitHubOAuthProvider, GitHubOAuthConfigurationError
from agent.infrastructure.sqlite_control_plane_repository import SqliteControlPlaneRepository
from local_agent_api_client import LocalAgentApiClient
import web_app


class GitHubIdentityManagementTests(unittest.TestCase):
    base='/api/control-plane/organizations/local-org/github-identities'

    def setUp(self):
        fixture.BrowserIdentityTests.setUp(self)
        self.repo.initialize_website_ownership('device-local-01')
        self.tokens={}
        for user,role in [('alice','operator'),('admin','admin'),('viewer','viewer'),('other-owner','owner')]:
            self.repo._connection.execute('INSERT INTO control_plane_principals VALUES(?,?,?,?)',(user,'local-org',role,'[]'))
        for user in ['local-owner','alice','admin','viewer','other-owner']:
            self.tokens[user]=self.repo.issue_identity_session(organization_id='local-org',user_id=user)['principal_token']
        self.calls=[];self.outage=False;self.callback_subject='43'
        def upstream(request):
            self.calls.append((request.method,request.url.path,'authorization' in request.headers))
            if self.outage:return httpx.Response(503)
            if request.method=='GET':
                users={'owner-gh':42,'alice-gh':43,'other-gh':44}
                ident=request.url.path.split('/')[-1]
                subject=int(ident) if ident.isdigit() else users.get(ident)
                if subject not in users.values():return httpx.Response(404)
                return httpx.Response(200,json={'id':subject,'login':next(k for k,v in users.items() if v==subject),'type':'User'})
            if request.url.host=='github.com':return httpx.Response(200,json={'access_token':'token-'+self.callback_subject})
            subject=int(json.loads(request.content)['access_token'].split('-')[-1])
            return httpx.Response(200,json={'app':{'client_id':'test-client'},'user':{'id':subject,'login':'renamed-account'}})
        self.provider=GitHubOAuthProvider(client_id='test-client',client_secret='test-secret',
            mapping_resolver=self.repo.github_identity,transport=httpx.MockTransport(upstream))
        self.addCleanup(self.provider.close)
        registry=web_app.ExternalIdentityProviderRegistry();registry.register(self.provider)
        p=patch.object(web_app,'control_plane_identity_registry',registry);p.start();self.addCleanup(p.stop)

    def auth(self,actor='local-owner',key='write'):
        return {**self.headers,'X-Control-Plane-Token':self.tokens[actor],'Idempotency-Key':key}

    def link(self,subject='43',user='alice',revision=0,actor='local-owner',key='link'):
        return self.client.put(self.base+'/'+subject,headers=self.auth(actor,key),json={'user_id':user,'expected_revision':revision})

    def test_lookup_link_login_unlink_and_relink_never_revives_sessions(self):
        r=self.client.get(self.base+'/lookup',params={'login':'alice-gh'},headers=self.auth())
        self.assertEqual(200,r.status_code,r.text);self.assertEqual('43',r.json()['data']['profile']['subject'])
        self.assertEqual(200,self.link().status_code)
        state=fixture.BrowserIdentityTests.start(self)
        callback=fixture.BrowserIdentityTests.complete(self,state)
        self.assertEqual('/?login=success',callback.headers['location'])
        status=fixture.BrowserIdentityTests.status(self)
        self.assertEqual('alice',status.json()['data']['session']['user_id'])
        token=self.client.cookies.get(web_app._BROWSER_SESSION_COOKIE)
        principal=self.repo.authenticate_identity_session(token)
        self.repo.claim_resource('session','private-history',principal,device_id='device-local-01')
        self.assertEqual(200,self.link(user=None,revision=1,key='unlink').status_code)
        self.assertIsNone(self.repo.authenticate_identity_session(token))
        self.assertFalse(fixture.BrowserIdentityTests.status(self).json()['data']['authenticated'])
        self.assertIsNotNone(self.repo.owned_resource('session','private-history',principal))
        denied=fixture.BrowserIdentityTests.complete(self,fixture.BrowserIdentityTests.start(self))
        self.assertEqual('/?login_error=external_identity_mapping_not_found',denied.headers['location'])
        self.assertEqual(200,self.link(revision=2,key='relink').status_code)
        self.assertIsNone(self.repo.authenticate_identity_session(token))
        self.assertEqual('alice',self.provider.authenticate({'access_token':'token-43'}).user_id)
        self.assertEqual('local-owner',self.repo.device_assignments('local-org')[0]['user_id'])
        self.assertTrue(all(not auth for method,_,auth in self.calls if method=='GET'))

    def test_member_hierarchy_foreign_org_and_identity_cannot_be_overwritten(self):
        self.assertEqual(403,self.link(user='local-owner',actor='admin').status_code)
        self.assertEqual(403,self.link(user='admin',actor='admin').status_code)
        self.assertEqual(200,self.link(actor='admin').status_code)
        self.assertEqual(409,self.link(user='viewer',revision=1,key='steal').status_code)
        self.assertEqual(403,self.link(actor='viewer').status_code)
        self.assertEqual(403,self.client.get(self.base,headers=self.auth('alice')).status_code)
        self.assertEqual(403,self.client.get(self.base.replace('local-org','foreign'),headers=self.auth()).status_code)
        self.repo._connection.execute("INSERT INTO control_plane_organizations VALUES('foreign','Foreign','foreign-owner','[]')")
        self.repo._connection.execute("INSERT INTO control_plane_principals VALUES('foreign-owner','foreign','owner','[]')")
        self.repo._connection.execute("INSERT INTO control_plane_github_identities VALUES('44','foreign','foreign-owner','private-login',0)")
        lookup=self.client.get(self.base+'/lookup',params={'login':'other-gh'},headers=self.auth())
        self.assertTrue(lookup.json()['data']['external_binding']);self.assertNotIn('foreign-owner',lookup.text)
        self.assertEqual(409,self.link(subject='44',user=None,key='foreign-unlink').status_code)
        self.assertEqual(409,self.link(subject='44',key='foreign-link').status_code)
        self.assertNotIn('private-login',self.client.get(self.base,headers=self.auth()).text)

    def test_last_login_owner_guard_also_applies_to_member_role_and_status(self):
        denied=self.link(subject='42',user=None,key='remove-owner')
        self.assertEqual(409,denied.status_code);self.assertEqual('last_linked_owner_required',denied.json()['detail']['code'])
        for role,status in [('viewer','active'),('owner','disabled')]:
            r=self.client.put('/api/control-plane/organizations/local-org/principals/local-owner',headers=self.auth(key=status),
                json={'role':role,'status':status,'expected_revision':0})
            self.assertEqual(409,r.status_code,r.text)
            self.assertEqual('last_linked_owner_required',r.json()['detail']['code'])
        self.assertEqual(200,self.link(subject='44',user='other-owner',key='second-owner').status_code)
        self.assertEqual(200,self.link(subject='42',user=None,key='unlink-owner-after').status_code)
        self.assertIsNone(self.repo.authenticate_identity_session(self.tokens['local-owner']))

    def test_unlink_between_provider_verification_and_issue_cannot_issue_stale_login(self):
        self.assertEqual(200,self.link().status_code)
        verified=self.provider.authenticate({'access_token':'token-43'})
        self.assertEqual(200,self.link(user=None,revision=1,key='unlink').status_code)
        self.assertEqual(200,self.link(revision=2,key='restore').status_code)
        with self.assertRaisesRegex(KeyError,'external_identity_mapping_changed'):
            self.repo.issue_external_identity_session(verified)
        with patch.object(self.provider,'authenticate',return_value=verified):
            r=self.client.post('/api/control-plane/auth/external-sessions',headers=self.headers,
                json={'provider_id':'github','assertion':{'access_token':'token-43'}})
        self.assertEqual(401,r.status_code,r.text)
        self.assertEqual('external_identity_mapping_changed',r.json()['detail']['code'])

    def test_revoked_admin_during_public_lookup_cannot_write(self):
        def lookup(**_):
            session=self.repo.authenticate_identity_session(self.tokens['admin'])
            self.repo.revoke_identity_session(session['session_id'])
            return {'subject':'43','login':'alice-gh'}
        with patch.object(self.provider,'public_user',side_effect=lookup):
            self.assertEqual(401,self.link(actor='admin').status_code)
        self.assertIsNone(self.repo.github_identity('43'))

    def test_separate_connections_serialize_login_with_unlink(self):
        self.assertEqual(200,self.link().status_code)
        verified=self.provider.authenticate({'access_token':'token-43'})
        other=SqliteControlPlaneRepository(self.repo.database_path)
        try:
            def issue():
                try:return other.issue_external_identity_session(verified)['principal_token']
                except KeyError:return None
            with ThreadPoolExecutor() as pool:
                issued=pool.submit(issue)
                removed=pool.submit(self.repo.update_github_identity,organization_id='local-org',principal_user_id='local-owner',
                    subject='43',user_id=None,expected_revision=1,idempotency_key='concurrent-unlink')
                self.assertEqual(200,removed.result()[0]);token=issued.result()
            if token:self.assertIsNone(self.repo.authenticate_identity_session(token))
            self.assertIsNone(self.repo.github_identity('43')['user_id'])
        finally:other.close()

    def test_untrusted_public_profile_cannot_change_the_requested_identity(self):
        for profile in [{'id':44,'login':'other-gh','type':'User'}, {'id':True,'login':'alice-gh','type':'User'},
                        {'id':43,'login':'alice-gh','type':'Organization'}, {'id':43,'login':'../unsafe','type':'User'}]:
            with self.subTest(profile=profile),patch.object(self.provider._client,'get',return_value=httpx.Response(200,json=profile)):
                self.assertEqual(503,self.link().status_code)
        self.assertIsNone(self.repo.github_identity('43'))

    def test_idempotency_stale_revision_and_outage_retry(self):
        self.assertEqual(200,self.link().status_code);calls=len(self.calls);self.outage=True
        replay=self.link();self.assertEqual(200,replay.status_code);self.assertTrue(replay.json()['idempotent_replay'])
        self.assertEqual(calls,len(self.calls))
        self.assertEqual(409,self.link(user='viewer').status_code)
        self.outage=False
        self.assertEqual(409,self.link(key='stale').status_code)
        self.outage=True
        self.assertEqual(503,self.link(subject='44',key='outage').status_code)
        self.assertIsNone(self.repo.github_identity('44'))
        self.assertEqual(200,self.link(user=None,revision=1,key='offline-unlink').status_code)

    def test_concurrent_competing_links_only_one_wins_and_is_durable(self):
        with ThreadPoolExecutor() as pool:
            replies=list(pool.map(lambda user:self.link(user=user,key=user),['alice','viewer']))
        self.assertEqual([200,409],sorted(r.status_code for r in replies))
        other=SqliteControlPlaneRepository(self.repo.database_path)
        try:
            self.assertEqual(1,other.github_identity('43')['revision'])
            audit=[e for e in other.read_audit('local-org',limit=100)['entries'] if e['resource']=='github_identity.update' and e['outcome']=='linked']
            self.assertEqual(1,len(audit));self.assertEqual('local-owner',audit[0]['actor'])
        finally:other.close()

    def test_strict_body_profile_failure_and_sdk(self):
        self.assertEqual(422,self.client.put(self.base+'/43',headers=self.auth(),json={'expected_revision':0}).status_code)
        self.assertEqual(422,self.client.put(self.base+'/43',headers=self.auth(),json={'user_id':'alice','expected_revision':True}).status_code)
        self.assertEqual(404,self.link(subject='999').status_code)
        self.assertEqual(422,self.link(subject='043').status_code)
        self.assertEqual(422,self.client.get(self.base+'/lookup',params={'login':'../admin'},headers=self.auth()).status_code)
        def transport(request):
            r=self.client.request(request.method,request.url.raw_path.decode(),headers=dict(request.headers),content=request.content)
            return httpx.Response(r.status_code,content=r.content,headers=dict(r.headers))
        with LocalAgentApiClient(transport=httpx.MockTransport(transport),principal_token=self.tokens['local-owner']) as sdk:
            self.assertEqual('43',sdk.control_plane_github_lookup('local-org',login='alice-gh')['data']['profile']['subject'])
            self.assertEqual(1,len(sdk.control_plane_github_identities('local-org')['data']['identities']))
            self.assertEqual('alice',sdk.control_plane_update_github_identity('local-org','43',user_id='alice',expected_revision=0,idempotency_key='sdk')['data']['identity']['user_id'])


class GitHubIdentityMigrationTests(unittest.TestCase):
    def test_import_is_atomic_idempotent_and_legacy_file_is_not_login_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);store=SqliteControlPlaneRepository(path/'runtime.sqlite3')
            try:
                with self.assertRaises(ValueError):
                    store.import_github_mappings({'42':{'organization_id':'local-org','user_id':'local-owner'},'43':{'organization_id':'local-org','user_id':'missing'}})
                self.assertFalse(store.github_mappings_initialized());self.assertIsNone(store.github_identity('42'))
                legacy=path/'identities.json';legacy.write_text(json.dumps({'42':{'organization_id':'local-org','user_id':'local-owner'}}))
                env={'GITHUB_OAUTH_CLIENT_ID':'test-client','GITHUB_OAUTH_CLIENT_SECRET':'test-secret','GITHUB_OAUTH_IDENTITY_MAP_FILE':str(legacy)}
                provider=GitHubOAuthProvider.from_environment(env,identity_store=store);provider.close()
                self.assertEqual('local-owner',store.github_identity('42')['user_id'])
                legacy.write_text('invalid file intentionally retained')
                provider=GitHubOAuthProvider.from_environment(env,identity_store=store);provider.close()
                env.pop('GITHUB_OAUTH_IDENTITY_MAP_FILE')
                provider=GitHubOAuthProvider.from_environment(env,identity_store=store);provider.close()
                self.assertFalse(store.import_github_mappings({'43':{'organization_id':'local-org','user_id':'local-owner'}}))
                self.assertIsNone(store.github_identity('43'))
            finally:store.close()
