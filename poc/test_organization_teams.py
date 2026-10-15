"""Team membership round trips through HTTP, SQLite, sessions and SDK."""
from concurrent.futures import ThreadPoolExecutor
import json
import unittest
from unittest.mock import patch
import httpx
import test_organization_members as members_fixture
from agent.infrastructure.sqlite_control_plane_repository import SqliteControlPlaneRepository
from local_agent_api_client import LocalAgentApiClient


class OrganizationTeamTests(unittest.TestCase):
    base = '/api/control-plane/organizations/local-org/teams'
    setUp = members_fixture.OrganizationMemberTests.setUp
    auth = members_fixture.OrganizationMemberTests.auth

    def create(self, team='ops', users=('alice',), actor='local-owner', key=None):
        return self.client.post(self.base, headers=self.auth(actor,key or 'create-'+team),
            json={'team_id':team,'name':team,'member_ids':list(users)})

    def update(self, team='ops', users=(), name='运营', revision=0, key='update', actor='local-owner'):
        return self.client.patch(self.base+'/'+team, headers=self.auth(actor,key),
            json={'name':name,'member_ids':list(users),'expected_revision':revision})

    def test_membership_roundtrip_existing_session_multiteam_and_no_resource_sharing(self):
        owner = self.repo.authenticate_identity_session(self.tokens['local-owner'])
        self.repo.claim_resource('session','owner-history',owner,device_id='device-local-01')
        for team in ('ops','support'):
            self.assertEqual(201,self.create(team,('alice','local-owner')).status_code)
        alice = self.repo.authenticate_identity_session(self.tokens['alice'])
        self.assertEqual(['ops','support'],alice['team_ids'])
        self.assertIsNone(self.repo.owned_resource('session','owner-history',alice))
        response=self.client.get('/api/agent/generic-supervised/owner-history',headers=self.auth('alice'))
        self.assertEqual(404,response.status_code)
        self.assertEqual('resource_not_found',response.json()['detail']['code'])
        self.assertEqual(200,self.update(users=('local-owner',)).status_code)
        self.assertEqual(['support'],self.repo.authenticate_identity_session(self.tokens['alice'])['team_ids'])
        self.assertEqual(['support'],json.loads(self.repo._connection.execute("SELECT team_ids_json FROM control_plane_principals WHERE user_id='alice'").fetchone()[0]))
        self.assertEqual('local-owner',self.repo.device_assignments('local-org')[0]['user_id'])
        self.assertEqual('operator',self.repo.authenticate_identity_session(self.tokens['alice'])['role'])

    def test_owner_admin_allowed_regular_users_and_other_org_denied(self):
        self.assertEqual(201,self.create(actor='admin',users=('admin','local-owner')).status_code)
        self.assertEqual(200,self.update(actor='admin').status_code)
        for actor in ('alice','viewer'):
            self.assertEqual(403,self.create('forbidden-'+actor,actor=actor).status_code)
            self.assertEqual(403,self.update(actor=actor,key=actor).status_code)
        self.assertEqual(403,self.client.get(self.base.replace('local-org','other'),headers=self.auth()).status_code)
        self.assertEqual(404,self.update(team='missing',key='missing').status_code)

    def test_invalid_members_duplicates_empty_team_and_disabled_members(self):
        for users in [('missing',),('alice','alice')]:
            self.assertEqual(422,self.create(users=users,key=str(users)).status_code)
        self.repo._connection.execute("INSERT INTO control_plane_organizations VALUES('other','Other','outsider','[]')")
        self.repo._connection.execute("INSERT INTO control_plane_principals VALUES('outsider','other','owner','[]')")
        self.assertEqual(422,self.create(users=('outsider',),key='foreign-member').status_code)
        self.repo._connection.execute("INSERT INTO control_plane_member_state VALUES('alice','disabled',1)")
        self.assertEqual(201,self.create().status_code)
        self.assertEqual(200,self.update(users=()).status_code)
        self.assertFalse(self.repo.member_active('alice'))
        self.assertEqual(409,self.create(key='duplicate').status_code)
        for body in ({'name':' '},{'name':'x','expected_revision':True},{'member_ids':None}):
            self.assertEqual(422,self.client.patch(self.base+'/ops',headers=self.auth(key=str(body)),json=body).status_code)

    def test_idempotency_conflict_persistence_and_actor_audit(self):
        self.assertEqual(201,self.create().status_code)
        self.assertTrue(self.create().json()['idempotent_replay'])
        self.assertEqual(200,self.update().status_code)
        self.assertTrue(self.update().json()['idempotent_replay'])
        self.assertEqual(409,self.update(name='different').status_code)
        self.assertEqual('team_conflict',self.update(key='stale').json()['detail']['code'])
        reopened=SqliteControlPlaneRepository(self.repo.database_path)
        try:
            team=next(t for t in reopened.list_teams('local-org') if t['team_id']=='ops')
            self.assertEqual(('运营',[],1),(team['name'],team['member_ids'],team['revision']))
            alice=next(p for p in reopened.list_principals('local-org') if p['user_id']=='alice')
            self.assertEqual([],alice['team_ids'])
            entries=[e for e in reopened.read_audit('local-org',limit=100)['entries'] if e['resource']=='team.update' and e['outcome']=='updated']
            self.assertEqual(1,len(entries)); self.assertEqual('local-owner',entries[0]['actor'])
            self.assertEqual(['alice'],entries[0]['metadata']['old_member_ids'])
        finally: reopened.close()

    def test_principal_creation_updates_team_and_invalidates_old_editor(self):
        self.assertEqual(201,self.create().status_code)
        result=self.client.post(self.base.replace('/teams','/principals'),headers=self.auth(key='new-member'),
            json={'user_id':'new-member','role':'viewer','team_ids':['ops','local-team']})
        self.assertEqual(201,result.status_code,result.text)
        team=next(t for t in self.repo.list_teams('local-org') if t['team_id']=='ops')
        self.assertEqual(['alice','new-member'],team['member_ids'])
        self.assertEqual(409,self.update().status_code)
        self.assertEqual(['local-team','ops'],next(p for p in self.repo.list_principals('local-org') if p['user_id']=='new-member')['team_ids'])

    def test_same_key_cannot_replay_update_for_another_team(self):
        self.create('ops'); self.create('support')
        self.assertEqual(200,self.update('ops').status_code)
        response=self.update('support')
        self.assertEqual(409,response.status_code)
        self.assertEqual('idempotency_conflict',response.json()['detail']['code'])
        self.assertEqual(['alice'],next(t for t in self.repo.list_teams('local-org') if t['team_id']=='support')['member_ids'])

    def test_two_connections_compare_revision_and_atomic_rollback(self):
        self.create()
        other=SqliteControlPlaneRepository(self.repo.database_path)
        def write(pair):
            store, name = pair
            try:
                return store.update_team(organization_id='local-org',principal_user_id='local-owner',
                    idempotency_key=name,request_hash=name,team_id='ops',name=name,member_ids=(),expected_revision=0)[0]
            except ValueError as error: return str(error)
        try:
            with ThreadPoolExecutor() as pool: replies=list(pool.map(write,[(self.repo,'a'),(other,'b')]))
            self.assertCountEqual([200,'team_conflict'],replies)
        finally: other.close()
        with patch.object(self.repo,'record_audit',side_effect=RuntimeError('disk error')):
            with self.assertRaises(RuntimeError):
                self.repo.update_team(organization_id='local-org',principal_user_id='local-owner',idempotency_key='rollback',
                    request_hash='rollback',team_id='ops',name='failed',member_ids=('alice',),expected_revision=1)
        self.assertEqual([],self.repo.authenticate_identity_session(self.tokens['alice'])['team_ids'])
        self.assertEqual(1,next(t for t in self.repo.list_teams('local-org') if t['team_id']=='ops')['revision'])

    def test_sdk_supplies_revision_and_checks_openapi(self):
        self.create()
        transport=httpx.MockTransport(lambda request:httpx.Response(
            (response:=self.client.request(request.method,str(request.url),headers=dict(request.headers),content=request.content)).status_code,
            content=response.content,headers={'Content-Type':'application/json'}))
        with LocalAgentApiClient(transport=transport,principal_token=self.tokens['local-owner']) as sdk:
            result=sdk.control_plane_update_team('local-org','ops',name='SDK',member_ids=[],expected_revision=0,
                principal_token=self.tokens['local-owner'],idempotency_key='sdk')
            self.assertEqual(1,result['data']['team']['revision'])


if __name__ == '__main__': unittest.main()
