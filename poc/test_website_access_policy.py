"""Isolated authorization counterexamples: no Runtime, device or network."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi import HTTPException
from starlette.requests import Request
from agent.infrastructure.sqlite_control_plane_repository import SqliteControlPlaneRepository
from agent.interfaces import website_access as access


class WebsiteAccessPolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'isolated.sqlite3'
        self.repo = SqliteControlPlaneRepository(self.path)
        self.addCleanup(self.repo.close)
        self.repo.initialize_website_ownership('device-local-01')
        self.repo._connection.execute("INSERT INTO control_plane_organizations VALUES('other-org','Other','other-owner','[]')")
        for user, org, role in [('alice','local-org','operator'), ('bob','local-org','admin'),
                                ('viewer','local-org','viewer'), ('other-owner','other-org','owner')]:
            self.repo._connection.execute('INSERT INTO control_plane_principals VALUES(?,?,?,?)', (user, org, role, '[]'))
        self.tokens = {}
        self.principals = {}
        for user in ('alice','bob','viewer','other-owner','local-owner'):
            org = 'other-org' if user == 'other-owner' else 'local-org'
            session = self.repo.issue_identity_session(organization_id=org,user_id=user)
            self.tokens[user] = session['principal_token']
            self.principals[user] = self.repo.authenticate_identity_session(session['principal_token'])
        self.repo.claim_resource('session','alice-private',self.principals['alice'],device_id='alice-device')
        self.repo.claim_resource('device','alice-device',self.principals['alice'],device_id='alice-device')

    def check(self, path, user=None, *, method='GET', params=None, headers=None):
        pairs = [(b'x-control-plane-token',self.tokens[user].encode())] if user else []
        pairs += [(k.encode(),v.encode()) for k,v in (headers or {}).items()]
        request = Request({'type':'http','method':method,'path':path,'query_string':b'',
                           'headers':pairs,'path_params':params or {},'route':SimpleNamespace(path=path)})
        asyncio.run(access.authorize(request,self.repo,lambda r: None,'device-local-01'))
        return request

    def denied(self, status, *args, **kwargs):
        with self.assertRaises(HTTPException) as raised:
            self.check(*args,**kwargs)
        self.assertEqual(status,raised.exception.status_code)

    def test_login_bootstrap_public_but_control_token_not_identity(self):
        for path in ('/','/api/session','/api/control-plane/auth/github/start',
                     '/api/control-plane/auth/github/callback','/api/control-plane/auth/browser-session',
                     '/api/control-plane/auth/browser-logout','/api/control-plane/auth/external-sessions'):
            self.check(path)
        self.denied(401,'/api/device',headers={'x-control-token':'public-bootstrap'})

    def test_owner_reads_own_tasks_admin_cannot_read_other_user(self):
        path='/api/agent/generic-supervised/{session_id}/events'
        params={'session_id':'alice-private'}
        self.check(path,'alice',params=params)
        for user in ('bob','local-owner','other-owner'):
            self.denied(404,path,user,params=params)
        self.denied(404,path,'alice',params={'session_id':'missing'})

    def test_cross_org_denied_and_viewer_cannot_write(self):
        path='/api/control-plane/organizations/{organization_id}/teams'
        self.check(path,'bob',method='POST',params={'organization_id':'local-org'})
        self.denied(403,path,'bob',params={'organization_id':'other-org'})
        self.denied(403,path,'viewer',method='POST',params={'organization_id':'local-org'})
        self.denied(403,'/api/agent/generic-supervised/start','viewer',method='POST')

    def test_device_scope_and_revoke_are_effective_before_dispatch(self):
        path='/api/device/{device_id}/directional-swipe'
        self.check(path,'alice',method='POST',params={'device_id':'alice-device'})
        self.denied(403,path,'bob',method='POST',params={'device_id':'alice-device'})
        self.repo.revoke_identity_session(self.principals['alice']['session_id'])
        self.denied(401,path,'alice',method='POST',params={'device_id':'alice-device'})

    def test_cookie_is_identity_and_invalid_header_cannot_fall_back(self):
        cookie=f'{access.PRINCIPAL_COOKIE}={self.tokens["alice"]}'
        self.check('/api/device',headers={'cookie':cookie})
        self.denied(401,'/api/device',headers={'cookie':cookie,'x-control-plane-token':'invalid'})

    def test_ownership_survives_reopen_and_migration_does_not_claim_future_ids(self):
        with self.assertRaises(ValueError):
            self.repo.claim_resource('session','alice-private',self.principals['bob'])
        reopened=SqliteControlPlaneRepository(self.path)
        self.addCleanup(reopened.close)
        reopened.initialize_website_ownership('different-device')
        self.assertIsNotNone(reopened.owned_resource('session','alice-private',self.principals['alice']))
        self.assertIsNone(reopened.owned_resource('device','different-device',self.principals['local-owner']))
        self.assertIsNone(reopened.owned_resource('session','future',self.principals['local-owner']))


if __name__ == '__main__': unittest.main()
