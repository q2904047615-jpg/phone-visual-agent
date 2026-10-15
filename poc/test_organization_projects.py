"""Project management remains a metadata operation separate from resource ownership."""
import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import test_organization_members as members_fixture
from agent.infrastructure.sqlite_control_plane_repository import SqliteControlPlaneRepository


class OrganizationProjectTests(unittest.TestCase):
    base='/api/control-plane/organizations/local-org/projects'
    setUp=members_fixture.OrganizationMemberTests.setUp
    auth=members_fixture.OrganizationMemberTests.auth

    def create(self, project='alpha', members=('alice',), actor='local-owner', key=None):
        return self.client.post(self.base,headers=self.auth(actor,key or 'create-'+project),json={
            'project_id':project,'name':project,'description':'initial','member_ids':list(members)})

    def update(self, project='alpha', revision=0, actor='local-owner', key='update', **body):
        body.setdefault('expected_revision',revision)
        return self.client.patch(self.base+'/'+project,headers=self.auth(actor,key),json=body)

    def test_create_update_archive_and_member_scope_does_not_share_resources(self):
        owner=self.repo.authenticate_identity_session(self.tokens['local-owner'])
        self.repo.claim_resource('session','private-history',owner,device_id='device-local-01')
        self.assertEqual(201,self.create().status_code)
        project=self.client.get(self.base,headers=self.auth('alice')).json()['data']['projects'][0]
        self.assertEqual(('alpha',['alice'],0),(project['name'],project['member_ids'],project['revision']))
        self.assertEqual(200,self.update(name='客户入驻',description='交付项目',member_ids=['local-owner'],status='active').status_code)
        self.assertEqual(200,self.update(revision=1,status='archived',name='客户入驻',description='交付项目',member_ids=['local-owner'],key='archive').status_code)
        item=self.client.get(self.base,headers=self.auth()) .json()['data']['projects'][0]
        self.assertEqual(('archived',2), (item['status'],item['revision']))
        self.assertEqual(404,self.client.get('/api/agent/generic-supervised/private-history',headers=self.auth('alice')).status_code)
        self.assertEqual('local-owner',self.repo.device_assignments('local-org')[0]['user_id'])

    def test_permissions_validation_and_cross_org(self):
        for actor in ('alice','viewer'):
            self.assertEqual(403,self.create('denied-'+actor,actor=actor).status_code)
            self.assertEqual(403,self.client.get(self.base.replace('local-org','foreign'),headers=self.auth(actor)).status_code)
        self.assertEqual(422,self.create('missing',('does-not-exist',),key='missing').status_code)
        self.assertEqual(422,self.create('duplicate',('alice','alice'),key='duplicate').status_code)
        self.assertEqual(201,self.create('empty',(),key='empty').status_code)
        self.assertEqual(422,self.update('empty',revision=0,key='bad-status',status='paused').status_code)
        self.assertEqual(422,self.update('empty',revision=0,key='empty-name',name=' ').status_code)
        self.assertEqual(404,self.update('missing',key='missing',name='Missing').status_code)

    def test_idempotency_binds_project_and_revision_conflict(self):
        self.assertEqual(201,self.create().status_code)
        self.assertTrue(self.create().json()['idempotent_replay'])
        self.assertEqual(201,self.create('beta',key='same-key').status_code)
        self.assertEqual(409,self.create('gamma',key='same-key').status_code)
        self.assertEqual(200,self.update(name='one').status_code)
        self.assertTrue(self.update(name='one').json()['idempotent_replay'])
        self.assertEqual(409,self.update(name='stale',key='stale').status_code)
        self.assertEqual('project_conflict',self.update(name='stale',key='stale').json()['detail']['code'])

    def test_persistence_audit_and_atomic_failure(self):
        self.assertEqual(201,self.create().status_code)
        with patch.object(self.repo,'record_audit',side_effect=RuntimeError('audit unavailable')):
            with self.assertRaises(RuntimeError): self.repo.update_project(organization_id='local-org',principal_user_id='local-owner',
                idempotency_key='rollback',request_hash='rollback',project_id='alpha',name='failed',description=None,status=None,member_ids=None,expected_revision=0)
        reopened=SqliteControlPlaneRepository(self.repo.database_path)
        try:
            item=reopened.list_projects('local-org')[0];self.assertEqual(('alpha','active',0), (item['name'],item['status'],item['revision']))
            audits=[e for e in reopened.read_audit('local-org',limit=100)['entries'] if e['resource']=='project.create' and e['outcome']=='created']
            self.assertEqual(1,len(audits));self.assertEqual('local-owner',audits[0]['actor'])
        finally:reopened.close()

    def test_concurrent_revision_only_one_update_wins(self):
        self.assertEqual(201,self.create().status_code)
        other=SqliteControlPlaneRepository(self.repo.database_path)
        def write(pair):
            store,name=pair
            try:return store.update_project(organization_id='local-org',principal_user_id='local-owner',idempotency_key=name,request_hash=name,project_id='alpha',name=name,description=None,status=None,member_ids=None,expected_revision=0)[0]
            except ValueError as error:return str(error)
        try:
            with ThreadPoolExecutor() as pool: replies=list(pool.map(write,[(self.repo,'one'),(other,'two')]))
            self.assertCountEqual([200,'project_conflict'],replies)
        finally:other.close()


if __name__=='__main__':unittest.main()
